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

sys.path.append(os.path.abspath(f"{_project_paths.UPSTREAM_ROOT}/gof_official"))
from utils.loss_utils import l1_loss, ssim
from utils.depth_utils import depth_to_normal
from gaussian_renderer import render as native_render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams

@register_method("gof")
class GOFWrapper(BaseMethod):
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
        self.trainCameras = self.scene.getTrainCameras().copy()
        self.viewpoint_stack = self.trainCameras.copy()
        
        self.highresolution_index = [i for i, c in enumerate(self.trainCameras) if c.image_width >= 800]
        self.gaussians.compute_3D_filter(cameras=self.trainCameras)
        
        self.last_n_gaussians = len(self.gaussians.get_xyz)

    def train_iteration(self, step):
        self.gaussians.update_learning_rate(step)
        if step % 1000 == 0:
            self.gaussians.oneupSHdegree()
            
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.trainCameras.copy()
            
        viewpoint_cam = self.viewpoint_stack.pop(random.randint(0, len(self.viewpoint_stack) - 1))
        
        if random.random() < 0.3 and self.dataset.sample_more_highres and len(self.highresolution_index) > 0:
            viewpoint_cam = self.trainCameras[self.highresolution_index[random.randint(0, len(self.highresolution_index) - 1)]]
            
        render_pkg = native_render(viewpoint_cam, self.gaussians, self.pipe, self.background, kernel_size=self.dataset.kernel_size)
        rendering = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        
        image = rendering[:3, :, :]
        gt_image = viewpoint_cam.original_image.cuda()
        
        Ll1 = l1_loss(image, gt_image)
        loss_target = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        distortion_map = rendering[8, :, :] if rendering.shape[0] > 8 else torch.zeros_like(image[0])
        loss_distortion = distortion_map.mean()
        
        depth = rendering[6, :, :] if rendering.shape[0] > 6 else torch.zeros_like(image[0])
        depth_normal, _ = depth_to_normal(viewpoint_cam, depth[None, ...])
        depth_normal = depth_normal.permute(2, 0, 1)
        render_normal = rendering[3:6, :, :] if rendering.shape[0] > 5 else torch.zeros_like(image)
        render_normal = F.normalize(render_normal, p=2, dim=0)
        c2w = (viewpoint_cam.world_view_transform.T).inverse()
        normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
        render_normal_world = normal2.reshape(3, *render_normal.shape[1:])
        normal_error = 1 - (render_normal_world * depth_normal).sum(dim=0)
        loss_normal = normal_error.mean()
        
        lambda_dist = self.opt.lambda_distortion if step >= self.opt.distortion_from_iter else 0.0
        lambda_norm = self.opt.lambda_depth_normal if step >= self.opt.depth_normal_from_iter else 0.0
        
        loss_parasitic = loss_distortion * lambda_dist + loss_normal * lambda_norm
        loss = loss_target + loss_parasitic

        grad_cos_sim = 0.0
        parasitic_ratio = 0.0

        if self.track_decoupling and step % 100 == 0 and loss_parasitic.item() > 0:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_target.backward(retain_graph=True)
            grad_target = self.gaussians._xyz.grad.clone() if self.gaussians._xyz.grad is not None else torch.zeros_like(self.gaussians._xyz)
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_parasitic.backward(retain_graph=True)
            grad_parasitic = self.gaussians._xyz.grad.clone() if self.gaussians._xyz.grad is not None else torch.zeros_like(self.gaussians._xyz)
            
            state = self.gaussians.optimizer.state.get(self.gaussians._xyz, None)
            if state is not None and "exp_avg_sq" in state:
                v_t = state["exp_avg_sq"]
                lr = self.gaussians.optimizer.param_groups[0]["lr"]
                u_t = (lr / (torch.sqrt(v_t) + 1e-8)) * grad_target
                u_p = (lr / (torch.sqrt(v_t) + 1e-8)) * grad_parasitic
            else:
                u_t = grad_target
                u_p = grad_parasitic
                
            norm_t = torch.norm(u_t, dim=1)
            norm_p = torch.norm(u_p, dim=1)
            valid_mask = (norm_t > 0) & (norm_p > 0)
            
            if valid_mask.any():
                grad_cos_sim = float(F.cosine_similarity(u_t[valid_mask], u_p[valid_mask], dim=1).mean())
            parasitic_ratio = float(norm_p.mean() / (norm_t.mean() + 1e-7))
            
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
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.05, self.scene.cameras_extent, size_threshold)
                    self.gaussians.compute_3D_filter(cameras=self.trainCameras)
                
                if step % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and step == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            if step % 100 == 0 and step > self.opt.densify_until_iter:
                if step < self.opt.iterations - 100:
                    self.gaussians.compute_3D_filter(cameras=self.trainCameras)

            if step < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        num_gaussians = self.gaussians.get_xyz.shape[0]
        metrics = {
            "loss": float(loss), "loss_l1": float(loss_target), "loss_ssim": float(loss_parasitic),
            "num_gaussians": int(num_gaussians), "delta_N": int(num_gaussians - self.last_n_gaussians),
            "peak_vram_GB": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
            "grad_cos_sim": float(grad_cos_sim), "parasitic_ratio": float(parasitic_ratio),
            "loss_distortion": float(loss_distortion), "loss_normal": float(loss_normal)
        }
        self.last_n_gaussians = num_gaussians
        
        histograms = {}
        if step % 1000 == 0:
            histograms["opacity"] = self.gaussians.get_opacity_with_3D_filter.clone().detach()
            scales = self.gaussians.get_scaling_with_3D_filter.clone().detach()
            histograms["scaling"] = scales
            scales_2d = scales[:, :2] if scales.shape[1] >= 2 else scales.unsqueeze(-1).expand(-1, 2)
            gamma = scales_2d.max(dim=-1)[0] / (scales_2d.min(dim=-1)[0] + 1e-7)
            histograms["anisotropy"] = gamma
            histograms["sh_dc_mag"] = self.gaussians._features_dc.detach().norm(dim=-1)

        return metrics, histograms
        
    def render(self, camera):
        with torch.no_grad():
            render_pkg = native_render(camera, self.gaussians, self.pipe, self.background, kernel_size=self.dataset.kernel_size)
        return {"image": render_pkg["render"][:3, :, :], "depth": render_pkg["render"][6, :, :] if render_pkg["render"].shape[0] > 6 else None}

    def save(self, save_dir, step):
        self.scene.save(step)

    def load(self, model_path, iteration):
        self.gaussians.load_ply(os.path.join(model_path, 'point_cloud', f'iteration_{iteration}', 'point_cloud.ply'))

    def get_spatial_centers(self):
        return self.gaussians._xyz

    def compute_physical_metrics(self, cameras=None):
        metrics = {}
        with torch.no_grad():
            scales = self.gaussians.get_scaling_with_3D_filter
            scales_2d = scales[:, :2] if scales.dim() > 1 and scales.shape[1] >= 2 else scales.unsqueeze(-1).expand(-1, 2)
            max_S, _ = torch.max(scales_2d, dim=1)
            min_S, _ = torch.min(scales_2d, dim=1)
            gamma = max_S / (min_S + 1e-7)
            
            metrics["gamma_median"] = float(torch.median(gamma))
            metrics["gamma_90th_percentile"] = float(torch.quantile(gamma, 0.90))
            metrics["scale_mean"] = float(torch.mean(scales_2d))
            metrics["alpha_mean"] = float(torch.mean(self.gaussians.get_opacity_with_3D_filter))
            
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
            opacities = self.gaussians.get_opacity_with_3D_filter.squeeze()
            scales = self.gaussians.get_scaling_with_3D_filter
            sigma_sq = (scales[:, :2].max(dim=1)[0].pow(2)) if scales.shape[1] >= 2 else scales.squeeze().pow(2)
            
            N_gaussians = xyz.shape[0]
            chunk_size = max(1, 30_000_000 // (N_gaussians + 1))
            for i in range(0, V, chunk_size):
                end = min(i + chunk_size, V)
                dist_sq = torch.cdist(query_points[i:end], xyz, p=2).pow(2)
                weights = torch.exp(-0.5 * dist_sq / (sigma_sq.unsqueeze(0) + 1e-7))
                densities[i:end] = torch.sum(weights * opacities.unsqueeze(0), dim=1)
            return densities

