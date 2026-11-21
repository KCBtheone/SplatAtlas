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

sys.path.append(os.path.abspath(f"{_project_paths.UPSTREAM_ROOT}/LightGaussian_official"))
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render as native_render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from prune import prune_list, calculate_v_imp_score
from torch.optim.lr_scheduler import ExponentialLR

@register_method("lightgaussian")
class LightGaussianWrapper(BaseMethod):
    def __init__(self, dataset_config, hyperparams):
        self.parser = ArgumentParser()
        self.lp = ModelParams(self.parser)
        self.op = OptimizationParams(self.parser)
        self.pp = PipelineParams(self.parser)
        
        self.parser.add_argument("--prune_iterations", nargs="+", type=int, default=[16000, 24000])
        self.parser.add_argument("--prune_percent", type=float, default=0.5)
        self.parser.add_argument("--v_pow", type=float, default=0.1)
        self.parser.add_argument("--prune_decay", type=float, default=0.8)
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
        self.gaussians.scheduler = ExponentialLR(self.gaussians.optimizer, gamma=0.97)

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.viewpoint_stack = self.scene.getTrainCameras().copy()
        
        self.last_n_gaussians = len(self.gaussians.get_xyz)
        self.pruning_ratio_current = 0.0

    def train_iteration(self, step):
        self.gaussians.update_learning_rate(step)
        if step % 1000 == 0:
            self.gaussians.oneupSHdegree()
            self.gaussians.scheduler.step()
            
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
            
        viewpoint_cam = self.viewpoint_stack.pop(random.randint(0, len(self.viewpoint_stack) - 1))
        
        render_pkg = native_render(viewpoint_cam, self.gaussians, self.pipe, self.background)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss_target = (1.0 - self.opt.lambda_dssim) * Ll1
        loss_parasitic = self.opt.lambda_dssim * (1.0 - ssim_value)
        
        loss = loss_target + loss_parasitic

        grad_cos_sim = 0.0
        parasitic_ratio = 0.0
        sti_value = 0.0

        if self.track_decoupling and step % 100 == 0:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_target.backward(retain_graph=True)
            grad_t = self.gaussians._xyz.grad.clone() if self.gaussians._xyz.grad is not None else torch.zeros_like(self.gaussians._xyz)
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_parasitic.backward(retain_graph=True)
            grad_p = self.gaussians._xyz.grad.clone() if self.gaussians._xyz.grad is not None else torch.zeros_like(self.gaussians._xyz)
            
            state = self.gaussians.optimizer.state.get(self.gaussians._xyz, None)
            if state is not None and "exp_avg_sq" in state:
                vt = state["exp_avg_sq"]
                eta_t = self.opt.position_lr_init 
                u_t = (eta_t / (torch.sqrt(vt) + 1e-8)) * grad_t
                u_p = (eta_t / (torch.sqrt(vt) + 1e-8)) * grad_p
                
                valid_mask = (torch.norm(u_t, dim=1) > 0) & (torch.norm(u_p, dim=1) > 0)
                if valid_mask.any():
                    cos_sim = F.cosine_similarity(u_t[valid_mask], u_p[valid_mask], dim=1)
                    grad_cos_sim = float(cos_sim.mean())
                    norm_t = torch.norm(u_t[valid_mask], dim=1)
                    norm_p = torch.norm(u_p[valid_mask], dim=1)
                    ratio = norm_p / (norm_t + norm_p + 1e-7)
                    parasitic_ratio = float(ratio.mean())
                    
                    conflict_mask = cos_sim < 0
                    if conflict_mask.any():
                        sti_value = float((ratio[conflict_mask] * torch.abs(cos_sim[conflict_mask])).mean())
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss.backward()
        else:
            loss.backward()

        prune_event = False
        with torch.no_grad():
            if step < self.opt.densify_until_iter:
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                if step > self.opt.densify_from_iter and step % self.opt.densification_interval == 0:
                    size_threshold = 20 if step > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)
                
                if step % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and step == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            if step in self.args.prune_iterations:
                prune_event = True
                gaussian_list, imp_list = prune_list(self.gaussians, self.scene, self.pipe, self.background)
                idx = self.args.prune_iterations.index(step)
                v_list = calculate_v_imp_score(self.gaussians, imp_list, self.args.v_pow)
                self.gaussians.prune_gaussians((self.args.prune_decay**idx) * self.args.prune_percent, v_list)

            if step < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        num_gaussians = self.gaussians.get_xyz.shape[0]
        delta_n = int(num_gaussians - self.last_n_gaussians)
        
        if prune_event:
            self.pruning_ratio_current = float(abs(delta_n) / (self.last_n_gaussians + 1e-7))
            
        metrics = {
            "loss": float(loss), "loss_l1": float(loss_target), "loss_ssim": float(loss_parasitic),
            "num_gaussians": int(num_gaussians), "delta_N": delta_n,
            "peak_vram_GB": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
            "prune_event": int(prune_event), "pruning_ratio_current": float(self.pruning_ratio_current)
        }
        
        if self.track_decoupling:
            metrics.update({
                "grad_cos_sim": float(grad_cos_sim),
                "parasitic_ratio": float(parasitic_ratio),
                "sti": float(sti_value)
            })
            
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
        return {"image": render_pkg["render"], "depth": render_pkg.get("depth", None)}

    def save(self, save_dir, step):
        self.scene.save(step)

    def load(self, model_path, iteration):
        ply_path = os.path.join(model_path, 'point_cloud', f'iteration_{iteration}', 'point_cloud.ply')
        self.gaussians.load_ply(ply_path)

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
            xyz, opacities = self.gaussians._xyz, torch.sigmoid(self.gaussians._opacity).squeeze()
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
