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

_AS_ROOT = os.path.abspath(f"{_project_paths.UPSTREAM_ROOT}/Analytic-Splatting")
assert os.path.isdir(_AS_ROOT), f"Analytic-Splatting repo not found at {_AS_ROOT}"
if _AS_ROOT not in sys.path:
    sys.path.insert(0, _AS_ROOT)
for _m in list(sys.modules):
    if _m == 'scene' or _m.startswith('scene.') \
       or _m == 'arguments' or _m.startswith('arguments.') \
       or _m == 'gaussian_renderer' or _m.startswith('gaussian_renderer.') \
       or _m == 'utils' or _m.startswith('utils.'):
        sys.modules.pop(_m, None)
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render as native_render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams

@register_method("analyticsplatting")
class AnalyticSplattingWrapper(BaseMethod):
    def __init__(self, dataset_config, hyperparams):
        self.parser = ArgumentParser()
        self.lp = ModelParams(self.parser)
        self.op = OptimizationParams(self.parser)
        self.pp = PipelineParams(self.parser)
        self.parser.add_argument("--filter3d", action="store_true")
        self.parser.add_argument("--dense", action="store_true")
        
        self.args = self.parser.parse_args([])
        self.args.source_path = dataset_config["source_path"]
        self.args.model_path = dataset_config["model_path"]
        self.args.eval = True
        self.args.resolution = dataset_config.get("resolution", 1)
        self.args.filter3d = True
        
        self.track_decoupling = hyperparams.get("track_decoupling", False)

        self.dataset = self.lp.extract(self.args)
        self.opt = self.op.extract(self.args)
        self.pipe = self.pp.extract(self.args)

        self.gaussians = GaussianModel(self.dataset.sh_degree)
        
        self.scene_scales = [1.0] if hyperparams.get("single_grid_training", False) else [1.0, 2.0, 4.0, 8.0]
        self.scene = Scene(self.dataset, self.gaussians, resolution_scales=self.scene_scales)
        self.gaussians.training_setup(self.opt)

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        self.trainCameras = []
        for scale in self.scene_scales:
            self.trainCameras += self.scene.getTrainCameras(scale).copy()
            
        self.gaussians.compute_3D_filter(cameras=self.trainCameras)
        self.viewpoint_stack = self.trainCameras.copy()
        
        self.last_n_gaussians = len(self.gaussians.get_xyz)

    def _get_effective_step(self, param, grad, group_lr):
        state = self.gaussians.optimizer.state.get(param)
        v = state.get("exp_avg_sq") if state is not None else torch.ones_like(param) * 1e-8
        return (group_lr / (torch.sqrt(v) + 1e-15)) * grad

    def train_iteration(self, step):
        self.gaussians.update_learning_rate(step)
        if step % 1000 == 0:
            self.gaussians.oneupSHdegree()
            
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.trainCameras.copy()
            
        viewpoint_cam = self.viewpoint_stack.pop(random.randint(0, len(self.viewpoint_stack) - 1))
        
        bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background
        
        render_pkg = native_render(viewpoint_cam, self.gaussians, self.pipe, bg, filter3d=self.args.filter3d)
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
        stats = {}

        if self.track_decoupling and step % 100 == 0:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_target.backward(retain_graph=True)
            grad_target_full = self.gaussians._xyz.grad.clone() if self.gaussians._xyz.grad is not None else torch.zeros_like(self.gaussians._xyz)
            
            grads_target_dict = {p: p.grad.clone() if p.grad is not None else torch.zeros_like(p) for group in self.gaussians.optimizer.param_groups for p in group["params"]}
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_parasitic.backward(retain_graph=True)
            grad_parasitic_full = self.gaussians._xyz.grad.clone() if self.gaussians._xyz.grad is not None else torch.zeros_like(self.gaussians._xyz)
            
            grads_parasitic_dict = {p: p.grad.clone() if p.grad is not None else torch.zeros_like(p) for group in self.gaussians.optimizer.param_groups for p in group["params"]}
            
            valid_mask = (torch.norm(grad_target_full, dim=1) > 0) & (torch.norm(grad_parasitic_full, dim=1) > 0)
            if valid_mask.any():
                grad_cos_sim = float(F.cosine_similarity(grad_target_full[valid_mask], grad_parasitic_full[valid_mask], dim=1).mean())
            parasitic_ratio = float(torch.norm(grad_parasitic_full, dim=1).mean() / (torch.norm(grad_target_full, dim=1).mean() + 1e-7))
            
            param_groups_map = {
                "spatial": [self.gaussians._xyz],
                "geometry": [self.gaussians._scaling, self.gaussians._rotation],
                "opacity": [self.gaussians._opacity],
                "appearance": [self.gaussians._features_dc, self.gaussians._features_rest]
            }

            for group_name, params in param_groups_map.items():
                U_T_list, U_P_list = [], []
                for p in params:
                    group_lr = 0.0
                    for opt_g in self.gaussians.optimizer.param_groups:
                        if any(p is q for q in opt_g["params"]):
                            group_lr = opt_g["lr"]
                            break
                    
                    ut = self._get_effective_step(p, grads_target_dict[p], group_lr)
                    up = self._get_effective_step(p, grads_parasitic_dict[p], group_lr)
                    U_T_list.append(ut.reshape(-1))
                    U_P_list.append(up.reshape(-1))
                
                U_T = torch.cat(U_T_list)
                U_P = torch.cat(U_P_list)
                
                if U_T.norm() > 0 and U_P.norm() > 0:
                    cos = float(F.cosine_similarity(U_T.unsqueeze(0), U_P.unsqueeze(0)))
                    r = float(U_P.norm() / (U_T.norm() + U_P.norm() + 1e-7))
                    ti = r * max(0.0, -cos)
                else:
                    ti = 0.0
                stats[f"sti_{group_name}"] = ti

            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss.backward()
        else:
            loss.backward()

        with torch.no_grad():
            if step < self.opt.densify_until_iter:
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                if step > self.opt.densify_from_iter and step % self.opt.densification_interval == 0:
                    size_threshold = 20 if step > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold, dense=self.args.dense)
                    self.gaussians.compute_3D_filter(cameras=self.trainCameras)
                
                if step % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and step == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            if step % 100 == 0 and step > self.opt.densify_until_iter:
                if step < self.opt.iterations - 100:
                    self.gaussians.compute_3D_filter(cameras=self.trainCameras)

            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)

        num_gaussians = self.gaussians.get_xyz.shape[0]
        metrics = {
            "loss": float(loss), "loss_l1": float(loss_target), "loss_ssim": float(loss_parasitic),
            "num_gaussians": int(num_gaussians), "delta_N": int(num_gaussians - self.last_n_gaussians),
            "peak_vram_GB": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
            "grad_cos_sim": float(grad_cos_sim), "parasitic_ratio": float(parasitic_ratio)
        }
        metrics.update(stats)
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
            if hasattr(self.gaussians, "filter_3D"):
                histograms["filter_3D"] = self.gaussians.filter_3D.clone().detach()

        return metrics, histograms
        
    def render(self, camera):
        with torch.no_grad():
            render_pkg = native_render(camera, self.gaussians, self.pipe, self.background, filter3d=self.args.filter3d)
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
            
            if hasattr(self.gaussians, "filter_3D"):
                metrics["filter_3d_mean"] = float(torch.mean(self.gaussians.filter_3D))
            
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
            
            if hasattr(self.gaussians, "filter_3D"):
                filter_sq = self.gaussians.filter_3D.squeeze().pow(2)
                sigma_sq = sigma_sq + filter_sq
            
            N_gaussians = xyz.shape[0]
            chunk_size = max(1, 30_000_000 // (N_gaussians + 1))
            for i in range(0, V, chunk_size):
                end = min(i + chunk_size, V)
                dist_sq = torch.cdist(query_points[i:end], xyz, p=2).pow(2)
                weights = torch.exp(-0.5 * dist_sq / (sigma_sq.unsqueeze(0) + 1e-7))
                densities[i:end] = torch.sum(weights * opacities.unsqueeze(0), dim=1)
            return densities
