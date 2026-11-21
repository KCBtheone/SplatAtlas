from core import paths as _project_paths

import os
import sys
import random
import torch
import numpy as np
import torch.nn.functional as F
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from core.registry import register_method
from core.base_method import BaseMethod

sys.path.append(os.path.abspath(f"{_project_paths.UPSTREAM_ROOT}/conegs"))
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render as native_render
from scene import Scene, GaussianModel
from utils.nerf_utils import get_num_rays_to_cast

@register_method("conegs")
class ConeGSWrapper(BaseMethod):
    def __init__(self, dataset_config, hyperparams):
        inference_only = hyperparams.get("inference_only", False)
        conegs_dir = os.path.abspath(f"{_project_paths.UPSTREAM_ROOT}/conegs")
        if not GlobalHydra().is_initialized():
            initialize_config_dir(config_dir=os.path.join(conegs_dir, 'configs'), version_base=None)
        self.cfg = compose(config_name="defaults")
        
        OmegaConf.set_struct(self.cfg, False)
        
        self.cfg.gaussian_model.source_path = dataset_config["source_path"]
        self.cfg.gaussian_model.model_path = dataset_config["model_path"]
        OmegaConf.set_struct(self.cfg, False)
        self.cfg.gaussian_model.eval = True
        if hasattr(self.cfg, 'dataset'):
            self.cfg.dataset.eval = True
        self.cfg.optimization.resolution = float(dataset_config.get("resolution", 1))
        if inference_only or hyperparams.get("explicit_training_grid", False):
            self.cfg.gaussian_model.resolution = dataset_config.get("resolution", 1)
        
        self.cfg.nerf_model.num_iters_pretrain = 20000
        self.cfg.nerf_model.nerf_train_during_3dgs = False
        self.cfg.optimization.use_preactivation_opacity_penalty = True
        self.cfg.optimization.opacity_penalty = 0.0002
        
        if "max_points" in hyperparams:
            self.cfg.optimization.max_points = hyperparams["max_points"]
        
        self.track_decoupling = hyperparams.get("track_decoupling", False)

        self.gaussians = GaussianModel(self.cfg.gaussian_model.sh_degree)
        self.scene = Scene(self.cfg.gaussian_model, self.gaussians,
                           stack_train_images=False if inference_only else self.cfg.nerf_model.stack_images)
        if not inference_only:
            self.gaussians.training_setup(self.cfg.optimization)
        
        bg_color = [1, 1, 1] if self.cfg.gaussian_model.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        self.nerf_model = None
        if not inference_only:
            from scene.nerf_model import NeRFModel
            self.nerf_model = NeRFModel(self.cfg.nerf_model)
            self.nerf_model.pretrain_model(self.cfg, self.gaussians, self.scene, self.background)

            self.nerf_model.prop_optimizer.zero_grad(set_to_none=True)
            self.nerf_model.model_optimizer.zero_grad(set_to_none=True)
            if hasattr(self.scene, "stacked_images"):
                del self.scene.stacked_images
            if hasattr(self.scene, "stacked_cam2worlds"):
                del self.scene.stacked_cam2worlds
            if hasattr(self.scene, "stacked_pix2cams"):
                del self.scene.stacked_pix2cams

            for i in range(len(self.scene.getTrainCameras())):
                self.scene.getTrainCameras()[i].original_image = self.scene.getTrainCameras()[i].original_image.cuda()
            for i in range(len(self.scene.getTestCameras())):
                self.scene.getTestCameras()[i].original_image = self.scene.getTestCameras()[i].original_image.cuda()

            num_rays_to_cast = get_num_rays_to_cast(len(self.gaussians._xyz), gaussian_percentage_increase=self.cfg.optimization.gaussian_percentage_increase, nerf_train_during_3dgs=self.cfg.nerf_model.nerf_train_during_3dgs)
            self.nerf_model.estimator_params["n_rays"] = num_rays_to_cast
            self.nerf_model.estimator_params["stratified"] = False
            self.nerf_model.estimator_params["requires_grad"] = False

        self.viewpoint_stack = self.scene.getTrainCameras().copy()
        self.last_n_gaussians = len(self.gaussians.get_xyz)

    def _get_u_step(self, params, grads):
        u_vecs = []
        for p, g in zip(params, grads):
            if g is None:
                continue
            state = self.gaussians.optimizer.state.get(p, None)
            if state is not None and "exp_avg_sq" in state:
                v = state["exp_avg_sq"]
                pg_lr = 1e-4
                for pg in self.gaussians.optimizer.param_groups:
                    if any(p is param for param in pg["params"]):
                        pg_lr = pg["lr"]
                        break
                u = (pg_lr / (torch.sqrt(v) + 1e-15)) * g
            else:
                u = g
            u_vecs.append(u.view(-1))
        if len(u_vecs) == 0:
            return torch.zeros(1, device="cuda")
        return torch.cat(u_vecs)

    def train_iteration(self, step):
        self.gaussians.update_learning_rate(step)
        if step % self.cfg.optimization.SH_increase_iter == 0:
            self.gaussians.oneupSHdegree()
            
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
            
        viewpoint_cam = self.viewpoint_stack.pop(random.randint(0, len(self.viewpoint_stack) - 1))
        gt_image = viewpoint_cam.original_image.cuda()
        
        bg = torch.rand((3), device="cuda") if self.cfg.optimization.random_background else self.background
        render_pkg = native_render(viewpoint_cam, self.gaussians, self.cfg.pipeline, bg)
        image = render_pkg["render"]
        
        l1_full = torch.abs((image - gt_image))
        ssim_full = ssim(image, gt_image)
        loss_target = (1.0 - self.cfg.optimization.lambda_dssim) * l1_full.mean()
        loss_ssim_part = self.cfg.optimization.lambda_dssim * (1.0 - ssim_full.mean())
        
        opacity = self.gaussians._opacity if self.cfg.optimization.use_preactivation_opacity_penalty else torch.abs(self.gaussians.get_opacity)
        loss_opacity = self.cfg.optimization.opacity_penalty * opacity.mean()
        
        loss = loss_target + loss_ssim_part + loss_opacity
        
        weights = (1.0 - self.cfg.optimization.densification_lambda_dssim) * l1_full.mean(0) + self.cfg.optimization.densification_lambda_dssim * (1.0 - ssim_full.mean(0))
        weights = torch.clamp_min(weights, 0.0)
        
        metrics = {}
        
        if self.track_decoupling and step % 100 == 0:
            semantic_groups = {
                "spatial": [self.gaussians._xyz],
                "geometry": [self.gaussians._scaling, self.gaussians._rotation],
                "opacity": [self.gaussians._opacity],
                "appearance": [self.gaussians._features_dc, self.gaussians._features_rest]
            }
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_target.backward(retain_graph=True)
            grad_target_dict = {}
            for k, params in semantic_groups.items():
                grad_target_dict[k] = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]
                
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_opacity.backward(retain_graph=True)
            grad_parasitic_dict = {}
            for k, params in semantic_groups.items():
                grad_parasitic_dict[k] = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]
                
            for k in semantic_groups.keys():
                u_t = self._get_u_step(semantic_groups[k], grad_target_dict[k])
                u_p = self._get_u_step(semantic_groups[k], grad_parasitic_dict[k])
                
                norm_t = torch.norm(u_t)
                norm_p = torch.norm(u_p)
                
                if norm_p > 1e-7 and norm_t > 1e-7:
                    s_group = F.cosine_similarity(u_t.unsqueeze(0), u_p.unsqueeze(0)).squeeze()
                    r_group = norm_p / (norm_t + norm_p + 1e-7)
                    ti = float(r_group * torch.clamp(-s_group, min=0.0))
                else:
                    ti = 0.0
                metrics[f"TI_{k}"] = ti
                
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss.backward()
        else:
            loss.backward()

        with torch.no_grad():
            if self.cfg.optimization.use_3dgs_densification and step < self.cfg.optimization.densify_until_iter:
                radii = render_pkg["radii"]
                visibility_filter = render_pkg["visibility_filter"]
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg.get("optimized_samples_index", 0))

                if step > self.cfg.optimization.densify_from_iter and step % 100 == 0:
                    size_threshold = 20 if step > 3000 else None
                    self.gaussians.densify_and_prune(0.0002, 0.005, self.scene.cameras_extent, size_threshold, max_points=self.cfg.optimization.max_points)
                
                if step % 3000 == 0:
                    self.gaussians.reset_opacity()

            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)

        if self.cfg.optimization.densify_from_iter < step and step < self.cfg.optimization.densify_until_iter and (self.cfg.optimization.gaussian_percentage_increase or self.cfg.optimization.max_points):
            num_rays_to_cast = get_num_rays_to_cast(len(self.gaussians._xyz), gaussian_percentage_increase=self.cfg.optimization.gaussian_percentage_increase, nerf_train_during_3dgs=self.cfg.nerf_model.nerf_train_during_3dgs)
            self.nerf_model.cast_rays_during_optimization(weights, num_rays_to_cast, viewpoint_cam, self.cfg, self.gaussians, self.scene, bg, step)

        with torch.no_grad():
            if self.cfg.optimization.densify_from_iter < step and step % 100 == 0:
                dead_mask = (self.gaussians.get_opacity <= 0.005).squeeze(-1)
                if step < self.cfg.optimization.densify_until_iter:
                    self.gaussians.prune_points(dead_mask)
                if self.cfg.optimization.gaussian_percentage_increase or self.cfg.optimization.max_points:
                    self.gaussians.initialize_new_points(use_cone_radius=True, max_points=self.cfg.optimization.max_points, opacity_init_value=self.cfg.optimization.opacity_init_value)

        num_gaussians = self.gaussians.get_xyz.shape[0]
        stashed_count = len(self.gaussians.stashed_xyz) if hasattr(self.gaussians, "stashed_xyz") and self.gaussians.stashed_xyz is not None else 0
        
        metrics.update({
            "loss": float(loss),
            "loss_l1": float(loss_target),
            "loss_ssim": float(loss_ssim_part),
            "num_gaussians": int(num_gaussians),
            "delta_N": int(num_gaussians - self.last_n_gaussians),
            "peak_vram_GB": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
            "stashed_points_count": int(stashed_count),
            "penalty_loss_ratio": float(loss_opacity / (loss + 1e-7))
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
            render_pkg = native_render(camera, self.gaussians, self.cfg.pipeline, self.background)
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
