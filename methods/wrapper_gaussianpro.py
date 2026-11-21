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

sys.path.append(os.path.abspath(f"{_project_paths.UPSTREAM_ROOT}/gaussianpro_official"))
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import load_pairs_relation
from utils.graphics_utils import check_geometric_consistency, depth_propagation, fov2focal
from gaussian_renderer import render as native_render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams

@register_method("gaussianpro")
class GaussianProWrapper(BaseMethod):
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

        self.opt.normal_loss = True
        self.opt.flatten_loss = True
        self.opt.sparse_loss = True
        self.opt.depth_loss = hyperparams.get("depth_loss", True)
        for name in (
            "dataset", "pair_path", "patch_size", "propagation_interval",
            "propagated_iteration_begin", "propagated_iteration_after",
            "depth_error_min_threshold", "depth_error_max_threshold",
        ):
            if name in hyperparams:
                setattr(self.opt, name, hyperparams[name])
        if self.opt.depth_loss and self.opt.propagation_interval <= 0:
            raise ValueError("GaussianPro propagation_interval must be positive")
        if self.opt.depth_loss and self.opt.propagated_iteration_begin >= self.opt.propagated_iteration_after:
            raise ValueError("GaussianPro propagation window must have a positive duration")
        self.opt.lambda_flatten = 100.0
        self.opt.lambda_l1_normal = 0.01
        self.opt.lambda_cos_normal = 0.01
        self.opt.lambda_sparse = 0.001

        self.gaussians = GaussianModel(self.dataset.sh_degree)
        self.scene = Scene(self.dataset, self.gaussians)
        self.gaussians.training_setup(self.opt)

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.train_cameras = self.scene.getTrainCameras()
        if (self.opt.depth_loss
                and not os.path.isdir(os.path.join(self.dataset.source_path, "sparse"))
                and os.path.isfile(os.path.join(self.dataset.source_path, "transforms_train.json"))):
            for camera in self.train_cameras:
                width, height = camera.image_width, camera.image_height
                camera.K = camera.K.new_tensor([
                    [fov2focal(camera.FoVx, width), 0, width / 2],
                    [0, fov2focal(camera.FoVy, height), height / 2],
                    [0, 0, 1],
                ])
        self.viewpoint_stack = list(range(len(self.train_cameras)))
        self.propagated_views = set()
        self.propagated_depths = {}
        self.propagation_pairs = None
        if self.opt.depth_loss and self.opt.dataset == "360":
            if not self.opt.pair_path:
                raise ValueError("GaussianPro 360 depth propagation requires pair_path")
            self.propagation_pairs = load_pairs_relation(self.opt.pair_path)
            if len(self.propagation_pairs) != len(self.train_cameras):
                raise ValueError("GaussianPro pair_path does not match the training camera count")
        
        self.last_n_gaussians = len(self.gaussians.get_xyz)

    def _propagation_neighbors(self, index):
        if self.propagation_pairs is not None:
            candidates = self.propagation_pairs[index]
        else:
            offsets = (-10, -5, 5, 10) if self.opt.dataset == "scannet" else (-2, -1, 1, 2)
            candidates = [index + offset for offset in offsets]
        return list(dict.fromkeys(
            other for other in candidates if 0 <= other < len(self.train_cameras) and other != index
        ))

    @torch.no_grad()
    def _propagate_view(self, camera_index, step, gt_image):
        opt = self.opt
        if (not opt.depth_loss
                or not opt.propagated_iteration_begin < step < opt.propagated_iteration_after
                or step % opt.propagation_interval != 0
                or camera_index in self.propagated_views):
            return 0
        neighbors = self._propagation_neighbors(camera_index)
        if len(neighbors) < 2:
            return 0

        camera = self.train_cameras[camera_index]
        projected = native_render(camera, self.gaussians, self.pipe, self.background, return_depth=True)
        depth, normal = depth_propagation(
            camera, projected["render_depth"], self.train_cameras, neighbors, opt.dataset, opt.patch_size,
        )
        self.propagated_depths[camera_index] = depth
        pose = camera.world_view_transform.T.inverse()
        votes = torch.zeros_like(depth, dtype=torch.int32)
        for index in neighbors:
            source = self.train_cameras[index]
            source_depth = self.propagated_depths.get(index)
            if source_depth is None:
                source_projection = native_render(
                    source, self.gaussians, self.pipe, self.background, return_depth=True,
                )
                source_neighbors = [camera_index] + [other for other in neighbors if other != index]
                source_depth, _ = depth_propagation(
                    source, source_projection["render_depth"], self.train_cameras,
                    source_neighbors, opt.dataset, opt.patch_size,
                )
                self.propagated_depths[index] = source_depth
            source_pose = source.world_view_transform.T.inverse()
            consistent, *_ = check_geometric_consistency(
                depth[None], camera.K[None], pose[None], source_depth[None],
                source.K[None], source_pose[None], thre1=2, thre2=0.01,
            )
            votes += consistent[0]

        valid = (votes >= 2) & torch.isfinite(depth) & (depth > 0) & (depth != 300)
        if camera.sky_mask is not None:
            valid &= camera.sky_mask.to(device=depth.device, dtype=torch.bool).reshape_as(depth)
        rotation = torch.as_tensor(camera.R.T, device=normal.device, dtype=normal.dtype)
        camera_normal = (rotation @ normal.reshape(-1, 3).T).reshape(3, *depth.shape)
        camera.normal = camera_normal.masked_fill(~valid.unsqueeze(0), -10)

        progress = (step - opt.propagated_iteration_begin) / (
            opt.propagated_iteration_after - opt.propagated_iteration_begin
        )
        threshold = opt.depth_error_max_threshold - progress * (
            opt.depth_error_max_threshold - opt.depth_error_min_threshold
        )
        relative_error = (depth - projected["render_depth"]).abs() / depth.clamp_min(1e-8)
        grow = valid & (relative_error > threshold)
        before = len(self.gaussians.get_xyz)
        if grow.sum() > 100 and grow[::8, ::8].any():
            self.gaussians.densify_from_depth_propagation(
                camera, depth.masked_fill(~grow, 300), grow, gt_image,
            )
        self.propagated_views.add(camera_index)
        return len(self.gaussians.get_xyz) - before

    def train_iteration(self, step):
        self.gaussians.update_learning_rate(step)
        if step % 1000 == 0:
            self.gaussians.oneupSHdegree()
            
        if not self.viewpoint_stack:
            self.viewpoint_stack = list(range(len(self.train_cameras)))
            
        camera_index = self.viewpoint_stack.pop(random.randint(0, len(self.viewpoint_stack) - 1))
        viewpoint_cam = self.train_cameras[camera_index]
        gt_image = viewpoint_cam.original_image.to(self.background.device)
        propagated_gaussians = self._propagate_view(camera_index, step, gt_image)
        
        render_pkg = native_render(viewpoint_cam, self.gaussians, self.pipe, self.background, return_normal=self.opt.normal_loss, return_opacity=True)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        
        opacity = render_pkg["render_opacity"]
        opacity_mask = opacity > 0.0
        if self.opt.depth_loss and step < self.opt.propagated_iteration_begin:
            opaque = opacity > 0.999
            if opaque.any():
                opacity_mask = opaque
        if not opacity_mask.any():
            opacity_mask = torch.ones_like(opacity_mask)
        opacity_mask = opacity_mask.unsqueeze(0).expand(3, -1, -1)
        
        Ll1 = l1_loss(image[opacity_mask], gt_image[opacity_mask])
        ssim_value = ssim(image, gt_image, mask=opacity_mask)
        loss_target = (1.0 - self.opt.lambda_dssim) * Ll1
        loss_target_ssim = self.opt.lambda_dssim * (1.0 - ssim_value)
        loss_target_total = loss_target + loss_target_ssim
        
        loss_parasitic = torch.tensor(0.0, device="cuda")
        loss_flatten_val = 0.0
        loss_normal_val = 0.0
        
        if self.opt.flatten_loss:
            scales = self.gaussians.get_scaling
            min_scale, _ = torch.min(scales, dim=1)
            min_scale = torch.clamp(min_scale, 0, 30)
            flatten_loss_tensor = torch.abs(min_scale).mean()
            loss_parasitic = loss_parasitic + self.opt.lambda_flatten * flatten_loss_tensor
            loss_flatten_val = float(flatten_loss_tensor)
            
        if self.opt.sparse_loss and visibility_filter.any():
            opacity = self.gaussians.get_opacity.clamp(1e-6, 1-1e-6)
            log_opacity = opacity * torch.log(opacity)
            log_one_minus_opacity = (1 - opacity) * torch.log(1 - opacity)
            sparse_loss_tensor = -1 * (log_opacity + log_one_minus_opacity)[visibility_filter].mean()
            loss_parasitic = loss_parasitic + self.opt.lambda_sparse * sparse_loss_tensor

        if self.opt.normal_loss and viewpoint_cam.normal is not None:
            rendered_normal = render_pkg["render_normal"]
            normal_gt = viewpoint_cam.normal.cuda()
            filter_mask = (normal_gt != -10)[0, :, :].to(torch.bool)
            if filter_mask.any():
                l1_normal = torch.abs(rendered_normal - normal_gt).sum(dim=0)[filter_mask].mean()
                cos_normal = (1.0 - torch.sum(rendered_normal * normal_gt, dim=0))[filter_mask].mean()
                normal_loss_tensor = self.opt.lambda_l1_normal * l1_normal + self.opt.lambda_cos_normal * cos_normal
                loss_parasitic = loss_parasitic + normal_loss_tensor
                loss_normal_val = float(normal_loss_tensor)

        loss = loss_target_total + loss_parasitic

        grad_cos_sim = 0.0
        parasitic_ratio = 0.0

        if self.track_decoupling and step % 100 == 0 and loss_parasitic.requires_grad:
            track_params = [
                self.gaussians._xyz, 
                self.gaussians._scaling, 
                self.gaussians._opacity, 
                self.gaussians._rotation
            ]
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_target_total.backward(retain_graph=True)
            grad_t_list = [p.grad.clone().view(p.shape[0], -1) if p.grad is not None else torch.zeros_like(p).view(p.shape[0], -1) for p in track_params]
            grad_target_cat = torch.cat(grad_t_list, dim=1)
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_parasitic.backward(retain_graph=True)
            grad_p_list = [p.grad.clone().view(p.shape[0], -1) if p.grad is not None else torch.zeros_like(p).view(p.shape[0], -1) for p in track_params]
            grad_parasitic_cat = torch.cat(grad_p_list, dim=1)
            
            valid_mask = (torch.norm(grad_target_cat, dim=1) > 0) & (torch.norm(grad_parasitic_cat, dim=1) > 0)
            if valid_mask.any():
                grad_cos_sim = float(F.cosine_similarity(grad_target_cat[valid_mask], grad_parasitic_cat[valid_mask], dim=1).mean())
            parasitic_ratio = float(torch.norm(grad_parasitic_cat, dim=1).mean() / (torch.norm(grad_target_cat, dim=1).mean() + 1e-7))
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            viewspace_point_tensor.grad = None
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
                
                if step % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and step == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)

        num_gaussians = self.gaussians.get_xyz.shape[0]
        metrics = {
            "loss": float(loss), 
            "loss_l1": float(loss_target), 
            "loss_ssim": float(loss_target_ssim),
            "loss_flatten": loss_flatten_val,
            "loss_normal": loss_normal_val,
            "propagated_gaussians": propagated_gaussians,
            "num_gaussians": int(num_gaussians), 
            "delta_N": int(num_gaussians - self.last_n_gaussians),
            "peak_vram_GB": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
            "grad_cos_sim": float(grad_cos_sim), 
            "parasitic_ratio": float(parasitic_ratio)
        }
        self.last_n_gaussians = num_gaussians
        
        histograms = {}
        if step % 1000 == 0:
            histograms["opacity"] = torch.sigmoid(self.gaussians._opacity).clone().detach()
            scales = torch.exp(self.gaussians._scaling).clone().detach()
            histograms["scaling"] = scales
            max_S, _ = torch.max(scales, dim=-1)
            min_S, _ = torch.min(scales, dim=-1)
            histograms["anisotropy"] = max_S / (min_S + 1e-7)
            histograms["sh_dc_mag"] = self.gaussians._features_dc.detach().norm(dim=-1)

        return metrics, histograms

    def render(self, camera):
        with torch.no_grad():
            render_pkg = native_render(camera, self.gaussians, self.pipe, self.background, return_depth=True)
        return {"image": render_pkg["render"], "depth": render_pkg["render_depth"]}

    def save(self, save_dir, step):
        self.scene.save(step)

    def load(self, model_path, iteration):
        self.gaussians.load_ply(os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))

    def get_spatial_centers(self):
        return self.gaussians._xyz

    def compute_physical_metrics(self, cameras=None):
        metrics = {}
        with torch.no_grad():
            scales = torch.exp(self.gaussians._scaling)
            
            max_S, _ = torch.max(scales, dim=1)
            min_S, _ = torch.min(scales, dim=1)
            gamma = max_S / (min_S + 1e-7)
            
            metrics["gamma_median"] = float(torch.median(gamma))
            metrics["gamma_90th_percentile"] = float(torch.quantile(gamma, 0.90))
            metrics["scale_mean"] = float(torch.mean(scales))
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
            sigma_sq = scales.max(dim=1)[0].pow(2)
            
            N_gaussians = xyz.shape[0]
            chunk_size = max(1, 30_000_000 // (N_gaussians + 1))
            for i in range(0, V, chunk_size):
                end = min(i + chunk_size, V)
                dist_sq = torch.cdist(query_points[i:end], xyz, p=2).pow(2)
                weights = torch.exp(-0.5 * dist_sq / (sigma_sq.unsqueeze(0) + 1e-7))
                densities[i:end] = torch.sum(weights * opacities.unsqueeze(0), dim=1)
            return densities
