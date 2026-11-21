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

_UPSTREAM = f'{_project_paths.UPSTREAM_ROOT}/SOGS_offy'
sys.path.insert(0, _UPSTREAM)
import os as _os
for _sub in ('utils', 'scene', 'gaussian_renderer', 'arguments', 'lpipsPyTorch', 'experiments'):
    _p = _os.path.join(_UPSTREAM, _sub)
    if _os.path.isdir(_p):
        sys.path.insert(0, _p)
del _os, _p, _sub, _UPSTREAM
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render as native_render
from scene import Scene, GaussianModel

class ConfigNode(dict):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            if isinstance(v, dict):
                self[k] = ConfigNode(**v)
            else:
                self[k] = v
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigNode' object has no attribute '{key}'")
    def __setattr__(self, key, value):
        self[key] = value

@register_method("sogs")
class SOGSWrapper(BaseMethod):
    def __init__(self, dataset_config, hyperparams):
        self.args = ConfigNode(
            source_path=dataset_config["source_path"],
            model_path=dataset_config["model_path"],
            eval=True,
            resolution=dataset_config.get("resolution", 1),
            images="images",
            data_device="cuda",
            white_background=False,
            sh_degree=3
        )
        
        self.track_decoupling = hyperparams.get("track_decoupling", False)

        self.opt = ConfigNode(
            iterations=30000,
            position_lr_init=0.00016,
            position_lr_final=0.0000016,
            position_lr_delay_mult=0.01,
            position_lr_max_steps=30000,
            feature_lr=0.0025,
            opacity_lr=0.05,
            scaling_lr=0.005,
            rotation_lr=0.001,
            percent_dense=0.01,
            lambda_dssim=0.2,
            densification_interval=100,
            opacity_reset_interval=3000,
            densify_from_iter=500,
            densify_until_iter=15000,
            densify_grad_threshold=0.0002,
            random_background=False,
            disable_xyz_log_activation=False
        )

        self.pipe = ConfigNode(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False
        )
        
        self.sorting = ConfigNode(
            enabled=True,
            weights=ConfigNode(xyz=1.0),
            improvement_break=0.001,
            shuffle=False,
            normalize=True,
            activated=True
        )
        
        self.neighbor_loss = ConfigNode(
            lambda_neighbor=1.0,
            weights=ConfigNode(opacity=1.0),
            blur=ConfigNode(kernel_size=5, sigma=1.0),
            loss_fn="mse",
            normalize=False,
            activated=True
        )

        self.run = ConfigNode(no_progress_bar=True, use_sh=True)

        self.gaussians = GaussianModel(self.args.sh_degree, disable_xyz_log_activation=self.opt.disable_xyz_log_activation)
        self.scene = Scene(self.args, self.gaussians)
        self.gaussians.training_setup(self.opt)

        if self.sorting.enabled:
            self.gaussians.prune_to_square_shape()
            self.gaussians.sort_into_grid(self.sorting, not self.run.no_progress_bar)

        bg_color = [1, 1, 1] if self.args.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.viewpoint_stack = self.scene.getTrainCameras().copy()
        
        self.last_n_gaussians = len(self.gaussians.get_xyz)

    def train_iteration(self, step):
        self.gaussians.update_learning_rate(step)
        if step % 1000 == 0 and self.run.use_sh:
            self.gaussians.oneupSHdegree()
            
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
            
        viewpoint_cam = self.viewpoint_stack.pop(random.randint(0, len(self.viewpoint_stack) - 1))
        
        bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background
        render_pkg = native_render(viewpoint_cam, self.gaussians, self.pipe, bg)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        
        Ll1 = l1_loss(image, gt_image)
        loss_target = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        if self.neighbor_loss.lambda_neighbor > 0:
            nb_losses = []
            attr_getter_fn = self.gaussians.get_activated_attr_flat if self.neighbor_loss.activated else self.gaussians.get_attr_flat
            weight_sum = sum(self.neighbor_loss.weights.values())
            for attr_name, attr_weight in self.neighbor_loss.weights.items():
                if attr_weight > 0:
                    nb_losses.append(self.gaussians.neighborloss_2d(attr_getter_fn(attr_name), self.neighbor_loss) * attr_weight / weight_sum)
            loss_parasitic = self.neighbor_loss.lambda_neighbor * sum(nb_losses)
        else:
            loss_parasitic = torch.tensor(0.0, device="cuda")

        loss = loss_target + loss_parasitic

        grad_cos_sim = 0.0
        parasitic_ratio = 0.0
        stats = {}

        if self.track_decoupling and step % 100 == 0 and loss_parasitic.item() > 0:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_target.backward(retain_graph=True)
            grad_target = self.gaussians._xyz.grad.clone() if self.gaussians._xyz.grad is not None else torch.zeros_like(self.gaussians._xyz)
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_parasitic.backward(retain_graph=True)
            grad_parasitic = self.gaussians._xyz.grad.clone() if self.gaussians._xyz.grad is not None else torch.zeros_like(self.gaussians._xyz)
            
            valid_mask = (torch.norm(grad_target, dim=1) > 0) & (torch.norm(grad_parasitic, dim=1) > 0)
            if valid_mask.any():
                grad_cos_sim = float(F.cosine_similarity(grad_target[valid_mask], grad_parasitic[valid_mask], dim=1).mean())
            parasitic_ratio = float(torch.norm(grad_parasitic, dim=1).mean() / (torch.norm(grad_target, dim=1).mean() + 1e-7))

            param_groups_map = {
                "spatial": [self.gaussians._xyz],
                "geometry": [self.gaussians._scaling, self.gaussians._rotation],
                "opacity": [self.gaussians._opacity],
                "appearance": [self.gaussians._features_dc, self.gaussians._features_rest],
            }

            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_target.backward(retain_graph=True)
            grads_target = {}
            for group_name, params in param_groups_map.items():
                group_grads = []
                for p in params:
                    if p.grad is not None:
                        state = self.gaussians.optimizer.state.get(p, {})
                        v_t = state.get("exp_avg_sq", torch.zeros_like(p))
                        lr = [pg["lr"] for pg in self.gaussians.optimizer.param_groups if pg["params"][0] is p][0]
                        u_t = (lr / (torch.sqrt(v_t) + 1e-15)) * p.grad.clone()
                        group_grads.append(u_t.reshape(-1))
                grads_target[group_name] = torch.cat(group_grads) if group_grads else None

            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_parasitic.backward(retain_graph=True)
            grads_parasitic = {}
            for group_name, params in param_groups_map.items():
                group_grads = []
                for p in params:
                    if p.grad is not None:
                        state = self.gaussians.optimizer.state.get(p, {})
                        v_t = state.get("exp_avg_sq", torch.zeros_like(p))
                        lr = [pg["lr"] for pg in self.gaussians.optimizer.param_groups if pg["params"][0] is p][0]
                        u_p = (lr / (torch.sqrt(v_t) + 1e-15)) * p.grad.clone()
                        group_grads.append(u_p.reshape(-1))
                grads_parasitic[group_name] = torch.cat(group_grads) if group_grads else None

            for group_name in param_groups_map:
                gt, gp = grads_target.get(group_name), grads_parasitic.get(group_name)
                if gt is not None and gp is not None and gt.norm() > 0 and gp.norm() > 0:
                    cos = float(F.cosine_similarity(gt.unsqueeze(0), gp.unsqueeze(0)))
                    r = float(gp.norm() / (gt.norm() + gp.norm() + 1e-7))
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
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)
                    
                    if self.sorting.enabled:
                        self.gaussians.prune_to_square_shape()
                        self.gaussians.sort_into_grid(self.sorting, not self.run.no_progress_bar)

                if step % self.opt.opacity_reset_interval == 0 or (self.args.white_background and step == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            if step < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        num_gaussians = self.gaussians.get_xyz.shape[0]
        metrics = {
            "loss": float(loss), 
            "loss_l1": float(loss_target), 
            "loss_ssim": float(loss_parasitic),
            "nb_loss": float(loss_parasitic),
            "num_gaussians": int(num_gaussians), 
            "delta_N": int(num_gaussians - self.last_n_gaussians),
            "peak_vram_GB": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
            "grad_cos_sim": float(grad_cos_sim), 
            "parasitic_ratio": float(parasitic_ratio)
        }
        if hasattr(self.gaussians, "grid_sidelen"):
            metrics["grid_sidelen"] = int(self.gaussians.grid_sidelen)
            
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

        return metrics, histograms
        
    def render(self, camera):
        with torch.no_grad():
            bg = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
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

