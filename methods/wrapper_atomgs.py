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

sys.path.append(os.path.abspath(f"{_project_paths.UPSTREAM_ROOT}/AtomGS_official"))
from utils.loss_utils import l1_loss, edge_aware_normal_loss, ms_ssim
from utils.image_utils import depth_to_normal
from gaussian_renderer import render as native_render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams

@register_method("atomgs")
class AtomGSWrapper(BaseMethod):
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
        
        self.dataset = self.lp.extract(self.args)
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
        self.gaussians.update_learning_rate(step)
        if step % 1000 == 0:
            self.gaussians.oneupSHdegree()
            
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
            
        viewpoint_cam = self.viewpoint_stack.pop(random.randint(0, len(self.viewpoint_stack) - 1))
        
        bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background
        render_pkg = native_render(viewpoint_cam, self.gaussians, self.pipe, bg)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        
        Ll1 = l1_loss(image, gt_image)
        L_ms_ssim = 1.0 - ms_ssim(image, gt_image)
        loss_photometric = (1.0 - self.opt.lambda_ms_ssim) * Ll1 + self.opt.lambda_ms_ssim * L_ms_ssim
        loss = loss_photometric
        
        Lnormal = torch.tensor(0.0, device="cuda")
        if step < self.opt.atom_proliferation_until:
            Lnormal = edge_aware_normal_loss(gt_image, depth_to_normal(render_pkg["mean_depth"], viewpoint_cam).permute(2, 0, 1))
            loss = loss + self.opt.lambda_normal * Lnormal

        grad_cos_sim = 0.0
        parasitic_ratio = 0.0

        if self.track_decoupling and step % 100 == 0 and step < self.opt.atom_proliferation_until:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_photometric.backward(retain_graph=True)
            g_T = {g['name']: g['params'][0].grad.clone() if g['params'][0].grad is not None else torch.zeros_like(g['params'][0]) for g in self.gaussians.optimizer.param_groups}
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_parasitic = self.opt.lambda_normal * Lnormal
            loss_parasitic.backward(retain_graph=True)
            g_P = {g['name']: g['params'][0].grad.clone() if g['params'][0].grad is not None else torch.zeros_like(g['params'][0]) for g in self.gaussians.optimizer.param_groups}
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            groups_map = {
                "spatial": ["xyz"],
                "geometry": ["scaling", "rotation"],
                "opacity": ["opacity"],
                "appearance": ["f_dc", "f_rest"]
            }
            
            sim_sum = 0.0
            ratio_sum = 0.0
            total_active = 0
            
            for g_name, keys in groups_map.items():
                U_T_g = []
                U_P_g = []
                for k in keys:
                    for group in self.gaussians.optimizer.param_groups:
                        if group['name'] == k:
                            p = group['params'][0]
                            state = self.gaussians.optimizer.state.get(p, None)
                            if state is not None and 'exp_avg_sq' in state:
                                v = state['exp_avg_sq']
                                lr = group['lr']
                                U_T_g.append((lr / (torch.sqrt(v) + 1e-15)) * g_T[k])
                                U_P_g.append((lr / (torch.sqrt(v) + 1e-15)) * g_P[k])
                
                if len(U_T_g) > 0:
                    U_T_vec = torch.cat([x.flatten() for x in U_T_g])
                    U_P_vec = torch.cat([x.flatten() for x in U_P_g])
                    
                    norm_P = torch.norm(U_P_vec)
                    if norm_P > 1e-7:
                        norm_T = torch.norm(U_T_vec)
                        cos_sim = float(F.cosine_similarity(U_T_vec.unsqueeze(0), U_P_vec.unsqueeze(0)).item())
                        ratio = float(norm_P / (norm_T + norm_P + 1e-7))
                        sim_sum += cos_sim
                        ratio_sum += ratio
                        total_active += 1
                        
            if total_active > 0:
                grad_cos_sim = sim_sum / total_active
                parasitic_ratio = ratio_sum / total_active
        else:
            loss.backward()

        with torch.no_grad():
            if step < self.opt.densify_until_iter:
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if step > self.opt.densify_from_iter and step % self.opt.densification_interval == 0:
                    self.gaussians.densify_and_prune(self.opt.clone_threshold, min(self.opt.split_threshold * step / self.opt.warm_up_until, self.opt.split_threshold), self.opt.prune_threshold)
                if step % self.opt.densification_interval == 0 and step < self.opt.atom_proliferation_until:
                    self.gaussians.atomize()
                if step % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and step == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            if step < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        num_gaussians = self.gaussians.get_xyz.shape[0]
        metrics = {
            "loss": float(loss),
            "loss_l1": float(Ll1),
            "loss_ssim": float(L_ms_ssim),
            "num_gaussians": int(num_gaussians),
            "delta_N": int(num_gaussians - self.last_n_gaussians),
            "peak_vram_GB": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
            "grad_cos_sim": float(grad_cos_sim),
            "parasitic_ratio": float(parasitic_ratio),
            "atom_scale": float(getattr(self.gaussians, "atom_scale", 0.0))
        }
        self.last_n_gaussians = num_gaussians
        
        histograms = {}
        if step % 1000 == 0:
            histograms["opacity"] = torch.sigmoid(self.gaussians._opacity).clone().detach()
            scales = torch.exp(self.gaussians._scaling).clone().detach()
            histograms["scaling"] = scales
            scales_2d = scales[:, :2] if scales.shape[1] >= 2 else scales
            gamma = scales_2d.max(dim=-1)[0] / (scales_2d.min(dim=-1)[0] + 1e-7)
            histograms["anisotropy"] = gamma
            histograms["sh_dc_mag"] = self.gaussians._features_dc.detach().norm(dim=-1)

        return metrics, histograms

    def render(self, camera):
        with torch.no_grad():
            render_pkg = native_render(camera, self.gaussians, self.pipe, self.background)
        return {"image": render_pkg["render"], "depth": render_pkg.get("mean_depth", None)}

    def save(self, save_dir, step):
        self.scene.save(step)

    def load(self, model_path, iteration):
        self.gaussians.load_ply(os.path.join(model_path, 'point_cloud', f'iteration_{iteration}', 'point_cloud.ply'))

    def get_spatial_centers(self):
        return self.gaussians._xyz

    def compute_physical_metrics(self, cameras=None):
        metrics = {}
        with torch.no_grad():
            raw_scales = self.gaussians._scaling
            scales = torch.exp(raw_scales)
            scales_2d = scales[:, :2] if scales.dim() > 1 and scales.shape[1] >= 2 else scales.unsqueeze(-1).expand(-1, 2)
            max_S, _ = torch.max(scales_2d, dim=1)
            min_S, _ = torch.min(scales_2d, dim=1)
            gamma = max_S / (min_S + 1e-7)
            
            metrics["gamma_median"] = float(torch.median(gamma))
            metrics["gamma_90th_percentile"] = float(torch.quantile(gamma, 0.90))
            metrics["scale_mean"] = float(torch.mean(scales_2d))
            metrics["alpha_mean"] = float(torch.mean(torch.sigmoid(self.gaussians._opacity)))
            
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
                normals = F.normalize(torch.stack([2*(x*z + w*y), 2*(y*z - w*x), 1-2*(x*x + y*y)], dim=-1), dim=1)
                
                max_cos, _ = torch.max(torch.abs(torch.matmul(normals, view_dirs.T)), dim=1)
                metrics["billboard_bias_ratio"] = float((max_cos > 0.90).float().mean())
        return metrics

    def evaluate_spatial_field(self, query_points: torch.Tensor, cameras=None) -> torch.Tensor:
        with torch.no_grad():
            V = query_points.shape[0]
            densities = torch.zeros(V, device="cuda")
            xyz = self.gaussians._xyz
            opacities = torch.sigmoid(self.gaussians._opacity).squeeze()
            scales = torch.exp(self.gaussians._scaling)
            sigma_sq = (scales[:, :2].max(dim=1)[0].pow(2)) if scales.shape[1] >= 2 else scales.squeeze().pow(2)
            
            N_gaussians = xyz.shape[0]
            chunk_size = max(1, 30_000_000 // (N_gaussians + 1))
            for i in range(0, V, chunk_size):
                end = min(i + chunk_size, V)
                dist_sq = torch.cdist(query_points[i:end], xyz, p=2).pow(2)
                weights = torch.exp(-0.5 * dist_sq / (sigma_sq.unsqueeze(0) + 1e-7))
                densities[i:end] = torch.sum(weights * opacities.unsqueeze(0), dim=1)
            return densities
