from core import paths as _project_paths

import os
import sys
import random
import torch
import numpy as np
import torch.nn.functional as F
from argparse import ArgumentParser

from core.registry import register_method
from core.base_method import BaseMethod

sys.path.append(os.path.abspath(f"{_project_paths.UPSTREAM_ROOT}/3dgsmcmc_official"))
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render as native_render
from scene import Scene, GaussianModel
from scene.gaussian_model import build_scaling_rotation
from arguments import ModelParams, PipelineParams, OptimizationParams

@register_method("3dgsmcmc")
class ThreeDGSMCMCWrapper(BaseMethod):
    def __init__(self, dataset_config, hyperparams):
        self.parser = ArgumentParser()
        self.lp = ModelParams(self.parser)
        self.op = OptimizationParams(self.parser)
        self.pp = PipelineParams(self.parser)
        self.args = self.parser.parse_args([])

        self.args.source_path = dataset_config["source_path"]
        self.args.model_path = dataset_config["model_path"]
        self.args.eval = True
        self.args.resolution = dataset_config.get("resolution", 1)
        self.track_decoupling = hyperparams.get("track_decoupling", False)
        
        scene_name = os.path.basename(os.path.normpath(self.args.source_path)).lower()
        indoor_scenes = ["bonsai", "counter", "kitchen", "room", "playroom", "drjohnson", "train"]
        is_indoor = "indoor" in self.args.source_path.lower() or scene_name in indoor_scenes
        self.args.cap_max = 2000000 if is_indoor else 4000000

        self.dataset = self.lp.extract(self.args)
        self.dataset.cap_max = self.args.cap_max
        self.opt = self.op.extract(self.args)
        self.pipe = self.pp.extract(self.args)

        self.gaussians = GaussianModel(self.dataset.sh_degree)
        self.scene = Scene(self.dataset, self.gaussians)
        self.gaussians.training_setup(self.opt)

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.viewpoint_stack = self.scene.getTrainCameras().copy()
        
        self.last_n_gaussians = len(self.gaussians.get_xyz)

    def train_iteration(self, step):
        xyz_lr = self.gaussians.update_learning_rate(step)
        
        if step % 1000 == 0:
            self.gaussians.oneupSHdegree()
            
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
            
        viewpoint_cam = self.viewpoint_stack.pop(random.randint(0, len(self.viewpoint_stack) - 1))
        
        bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background
        render_pkg = native_render(viewpoint_cam, self.gaussians, self.pipe, bg)
        image = render_pkg["render"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        
        loss_target = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim_value)
        loss_parasitic = self.opt.opacity_reg * torch.abs(self.gaussians.get_opacity).mean() + self.opt.scale_reg * torch.abs(self.gaussians.get_scaling).mean()
        loss = loss_target + loss_parasitic

        sti_metrics = {
            "sti_spatial": 0.0,
            "sti_geometry": 0.0,
            "sti_opacity": 0.0,
            "sti_appearance": 0.0
        }

        if self.track_decoupling and step % 100 == 0:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_target.backward(retain_graph=True)
            grads_t = {}
            for group in self.gaussians.optimizer.param_groups:
                name = group["name"]
                p = group['params'][0]
                if p.grad is not None:
                    grads_t[name] = p.grad.clone()
                else:
                    grads_t[name] = torch.zeros_like(p)
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_parasitic.backward(retain_graph=True)
            grads_p = {}
            states_v = {}
            lrs = {}
            epsilons = {}
            for group in self.gaussians.optimizer.param_groups:
                name = group["name"]
                p = group['params'][0]
                if p.grad is not None:
                    grads_p[name] = p.grad.clone()
                else:
                    grads_p[name] = torch.zeros_like(p)
                
                state = self.gaussians.optimizer.state.get(p, {})
                states_v[name] = state.get("exp_avg_sq", torch.zeros_like(p))
                lrs[name] = group["lr"]
                epsilons[name] = group.get("eps", 1e-15)

            semantic_groups = {
                "spatial": ["xyz"],
                "geometry": ["scaling", "rotation"],
                "opacity": ["opacity"],
                "appearance": ["f_dc", "f_rest"]
            }

            for sg_name, param_names in semantic_groups.items():
                u_t_list = []
                u_p_list = []
                for pname in param_names:
                    if pname in grads_t:
                        gt = grads_t[pname]
                        gp = grads_p[pname]
                        vt = states_v[pname]
                        lr = lrs[pname]
                        eps = epsilons[pname]
                        
                        denom = torch.sqrt(vt) + eps
                        ut = torch.nan_to_num((lr / denom) * gt, nan=0.0, posinf=0.0, neginf=0.0)
                        up = torch.nan_to_num((lr / denom) * gp, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        N = ut.shape[0]
                        u_t_list.append(ut.view(N, -1))
                        u_p_list.append(up.view(N, -1))
                
                if len(u_t_list) > 0:
                    M_UT = torch.cat(u_t_list, dim=1)
                    M_UP = torch.cat(u_p_list, dim=1)
                    
                    norm_ut = torch.norm(M_UT, dim=1)
                    norm_up = torch.norm(M_UP, dim=1)
                    
                    valid_mask = (norm_ut > 0) & (norm_up > 0)
                    
                    if valid_mask.any():
                        cos_sim = F.cosine_similarity(M_UT[valid_mask], M_UP[valid_mask], dim=1)
                        share = norm_up[valid_mask] / (norm_ut[valid_mask] + norm_up[valid_mask] + 1e-7)
                        ti = share * torch.clamp(-cos_sim, min=0.0)
                        sti_metrics[f"sti_{sg_name}"] = float(ti.mean().item())
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss.backward()
        else:
            loss.backward()

        with torch.no_grad():
            if self.opt.densify_from_iter < step < self.opt.densify_until_iter and step % self.opt.densification_interval == 0:
                dead_mask = (self.gaussians.get_opacity <= 0.005).squeeze(-1)
                self.gaussians.relocate_gs(dead_mask=dead_mask)
                self.gaussians.add_new_gs(cap_max=self.dataset.cap_max)

            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)

            L = build_scaling_rotation(self.gaussians.get_scaling, self.gaussians.get_rotation)
            actual_covariance = L @ L.transpose(1, 2)
            
            def op_sigmoid(x, k=100, x0=0.995):
                return 1.0 / (1.0 + torch.exp(-k * (x - x0)))
            
            noise = torch.randn_like(self.gaussians._xyz) * op_sigmoid(1.0 - self.gaussians.get_opacity) * self.opt.noise_lr * xyz_lr
            noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
            self.gaussians._xyz.add_(noise)

        num_gaussians = self.gaussians.get_xyz.shape[0]
        metrics = {
            "loss": float(loss), 
            "loss_l1": float(loss_target), 
            "loss_ssim": float(loss_parasitic),
            "num_gaussians": int(num_gaussians), 
            "delta_N": int(num_gaussians - self.last_n_gaussians),
            "peak_vram_GB": float(torch.cuda.max_memory_allocated() / (1024 ** 3))
        }
        metrics.update(sti_metrics)
        
        self.last_n_gaussians = num_gaussians
        
        histograms = {}
        if step % 1000 == 0:
            opacity = self.gaussians.get_opacity.clone().detach()
            scales = self.gaussians.get_scaling.clone().detach()
            
            histograms["opacity"] = opacity
            histograms["scaling"] = scales
            
            scales_2d = scales[:, :2] if scales.shape[1] >= 2 else scales
            gamma = scales_2d.max(dim=-1)[0] / (scales_2d.min(dim=-1)[0] + 1e-7)
            histograms["anisotropy"] = gamma
            
            histograms["sh_dc_mag"] = self.gaussians._features_dc.detach().norm(dim=-1)
            
            mcmc_modulation = 1.0 / (1.0 + torch.exp(-100.0 * ((1.0 - opacity) - 0.995)))
            histograms["mcmc_noise_modulation"] = mcmc_modulation.squeeze(-1)

        return metrics, histograms

    def render(self, camera):
        with torch.no_grad():
            bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
            render_pkg = native_render(camera, self.gaussians, self.pipe, bg)
        return {"image": render_pkg["render"], "depth": render_pkg.get("depth", None)}

    def save(self, save_dir, step):
        self.scene.save(step)

    def load(self, model_path, iteration):
        self.gaussians.load_ply(os.path.join(model_path, 'point_cloud', f'iteration_{iteration}', 'point_cloud.ply'))

    def get_spatial_centers(self):
        return self.gaussians._xyz

    def compute_physical_metrics(self, cameras=None):
        metrics = {}
        with torch.no_grad():
            scales = self.gaussians.get_scaling
            scales_2d = scales[:, :2] if scales.dim() > 1 and scales.shape[1] >= 2 else scales.unsqueeze(-1).expand(-1, 2)
            
            max_S, _ = torch.max(scales_2d, dim=1)
            min_S, _ = torch.min(scales_2d, dim=1)
            gamma = max_S / (min_S + 1e-7)
            
            metrics["gamma_median"] = float(torch.median(gamma))
            metrics["gamma_90th_percentile"] = float(torch.quantile(gamma, 0.90))
            metrics["scale_mean"] = float(torch.mean(scales_2d))
            metrics["alpha_mean"] = float(torch.mean(self.gaussians.get_opacity))
            
            dc, rest = self.gaussians._features_dc, self.gaussians._features_rest
            if rest is not None and rest.shape[1] > 0:
                metrics["sh_energy_ratio"] = float(rest.norm(dim=-1).mean() / (dc.norm(dim=-1).mean() + 1e-7))
                
            if cameras is not None and len(cameras) > 0:
                view_dirs = []
                for c in cameras:
                    view_dirs.append(c.world_view_transform[:3, 2].tolist())
                view_dirs = F.normalize(torch.tensor(view_dirs, dtype=torch.float32, device="cuda"), dim=1)

                rots = F.normalize(self.gaussians._rotation.clone(), dim=1)
                w, x, y, z = rots.unbind(dim=-1)
                normals = F.normalize(torch.stack([2.0*(x*z + w*y), 2.0*(y*z - w*x), 1.0-2.0*(x*x + y*y)], dim=-1), dim=1)
                
                max_cos, _ = torch.max(torch.abs(torch.matmul(normals, view_dirs.T)), dim=1)
                metrics["billboard_bias_ratio"] = float((max_cos > 0.90).float().mean())

        return metrics

    def evaluate_spatial_field(self, query_points: torch.Tensor, cameras=None) -> torch.Tensor:
        with torch.no_grad():
            V = query_points.shape[0]
            densities = torch.zeros(V, device="cuda")
            xyz, opacities = self.gaussians._xyz, self.gaussians.get_opacity.squeeze(-1)
            scales = self.gaussians.get_scaling
            sigma_sq = (scales[:, :2].max(dim=1)[0].pow(2)) if scales.shape[1] >= 2 else scales.squeeze(-1).pow(2)
            
            N_gaussians = xyz.shape[0]
            chunk_size = max(1, 30_000_000 // (N_gaussians + 1))
            for i in range(0, V, chunk_size):
                end = min(i + chunk_size, V)
                dist_sq = torch.cdist(query_points[i:end], xyz, p=2).pow(2)
                weights = torch.exp(-0.5 * dist_sq / (sigma_sq.unsqueeze(0) + 1e-7))
                densities[i:end] = torch.sum(weights * opacities.unsqueeze(0), dim=1)
            return densities
