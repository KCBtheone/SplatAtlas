from core import paths as _project_paths

import os
import json
import sys
import random
import torch
import numpy as np
import torch.nn.functional as F
from argparse import ArgumentParser

from core.registry import register_method
from core.base_method import BaseMethod

sys.path.insert(0, f"{_project_paths.UPSTREAM_ROOT}/Octree-GS_official")
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render as native_render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams

@register_method("octree_gs")
class OctreeGSWrapper(BaseMethod):
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
        self.args.images = dataset_config.get("images", "images")
        if self.args.images == "images":
            self.args.ds = 1
        elif self.args.images.startswith("images_") and self.args.images[7:].isdigit():
            self.args.ds = int(self.args.images[7:])
        else:
            raise ValueError("Octree-GS upstream requires images or images_<downscale> folders")

        self.reference_preset = hyperparams.get("reference_preset", "nerfbaselines")
        if self.reference_preset not in ("native", "nerfbaselines"):
            raise ValueError("reference_preset must be native or nerfbaselines")
        synthetic = (not os.path.isdir(os.path.join(self.args.source_path, "sparse"))
                     and os.path.isfile(os.path.join(self.args.source_path, "transforms_train.json")))
        dataset_parts = os.path.normpath(self.args.source_path).lower().split(os.sep)
        if synthetic:
            self.args.white_background = True

        if self.reference_preset == "nerfbaselines":
            self.args.appearance_dim = 0
            self.args.base_layer = 10
            self.args.visible_threshold = -1.0
            self.args.progressive = True
            if any(x in dataset_parts for x in ("tnt", "tanksandtemples", "deepblending")):
                self.args.visible_threshold = 0.9

        config_path = os.path.join(self.args.model_path, "anchor_gs_config.json")
        if os.path.isfile(config_path):
            with open(config_path) as config_file:
                saved_config = json.load(config_file)
            for key, value in saved_config["args"].items():
                if key not in ("source_path", "model_path", "images", "resolution") and hasattr(self.args, key):
                    setattr(self.args, key, value)
            self.reference_preset = saved_config["reference_preset"]
        for key, value in hyperparams.items():
            if hasattr(self.args, key):
                setattr(self.args, key, value)
        if "white_background" in dataset_config:
            self.args.white_background = dataset_config["white_background"]
        self.track_decoupling = hyperparams.get("track_decoupling", False)

        self.dataset = self.lp.extract(self.args)
        self.opt = self.op.extract(self.args)
        self.pipe = self.pp.extract(self.args)

        self.gaussians = GaussianModel(
            self.dataset.feat_dim, self.dataset.n_offsets, self.dataset.fork, self.dataset.use_feat_bank, self.dataset.appearance_dim, 
            self.dataset.add_opacity_dist, self.dataset.add_cov_dist, self.dataset.add_color_dist, self.dataset.add_level, 
            self.dataset.visible_threshold, self.dataset.dist2level, self.dataset.base_layer, self.dataset.progressive, self.dataset.extend
        )
        import logging; self.scene = Scene(self.dataset, self.gaussians, shuffle=False, resolution_scales=self.dataset.resolution_scales, logger=logging.getLogger("dummy"))
        self.gaussians.training_setup(self.opt)
        self.last_step = self.opt.iterations
        self.gaussians.set_coarse_interval(self.opt.coarse_iter, self.opt.coarse_factor)

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.viewpoint_stack = self.scene.getTrainCameras().copy()
        
        self.last_n_gaussians = 0

    def train_iteration(self, step):
        self.gaussians.update_learning_rate(step)
            
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
            
        viewpoint_cam = self.viewpoint_stack.pop(random.randint(0, len(self.viewpoint_stack) - 1))
        
        self.gaussians.set_anchor_mask(viewpoint_cam.camera_center, step, viewpoint_cam.resolution_scale)
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, self.gaussians, self.pipe, self.background)
        retain_grad = (step < self.opt.update_until and step >= 0)
        
        render_pkg = native_render(viewpoint_cam, self.gaussians, self.pipe, self.background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
        
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        offset_selection_mask = render_pkg.get("selection_mask", torch.zeros_like(visibility_filter))
        scaling = render_pkg.get("scaling", torch.empty(0))
        opacity = render_pkg.get("neural_opacity", torch.empty(0))
        
        gt_image = viewpoint_cam.original_image.cuda()
        
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        
        loss_target = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim_value)
        scaling_reg = scaling.prod(dim=1).mean() if scaling.shape[0] > 0 else torch.tensor(0.0, device="cuda")
        loss_parasitic = 0.01 * scaling_reg
        loss = loss_target + loss_parasitic

        grad_cos_sim = 0.0
        parasitic_ratio = 0.0

        if self.track_decoupling and step % 100 == 0 and self.gaussians._anchor.grad is not None:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_target.backward(retain_graph=True)
            grad_target = self.gaussians._anchor.grad.clone() if self.gaussians._anchor.grad is not None else torch.zeros_like(self.gaussians._anchor)
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_parasitic.backward(retain_graph=True)
            grad_parasitic = self.gaussians._anchor.grad.clone() if self.gaussians._anchor.grad is not None else torch.zeros_like(self.gaussians._anchor)
            
            valid_mask = (torch.norm(grad_target, dim=1) > 0) & (torch.norm(grad_parasitic, dim=1) > 0)
            if valid_mask.any():
                grad_cos_sim = float(F.cosine_similarity(grad_target[valid_mask], grad_parasitic[valid_mask], dim=1).mean())
            parasitic_ratio = float(torch.norm(grad_parasitic, dim=1).mean() / (torch.norm(grad_target, dim=1).mean() + 1e-7))
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss.backward()
        else:
            loss.backward()

        with torch.no_grad():
            if step < self.opt.update_until and step > self.opt.start_stat:
                self.gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                if self.opt.update_anchor and step > self.opt.update_from and step % self.opt.update_interval == 0:
                    self.gaussians.adjust_anchor(
                        iteration=step,
                        check_interval=self.opt.update_interval, 
                        success_threshold=self.opt.success_threshold,
                        grad_threshold=self.opt.densify_grad_threshold, 
                        update_ratio=self.dataset.update_ratio,
                        extra_ratio=self.dataset.extra_ratio,
                        extra_up=self.dataset.extra_up,
                        min_opacity=self.opt.min_opacity
                    )
            elif step == self.opt.update_until:
                if hasattr(self.gaussians, 'opacity_accum'):
                    del self.gaussians.opacity_accum
                    del self.gaussians.offset_gradient_accum
                    del self.gaussians.offset_denom
                torch.cuda.empty_cache()

            if step < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.last_step = step
        num_gaussians = int(visibility_filter.sum().item()) if visibility_filter.shape[0] > 0 else 0
        num_anchors = self.gaussians.get_anchor.shape[0]

        metrics = {
            "loss": float(loss), "loss_l1": float(loss_target), "loss_ssim": float(1.0 - ssim_value),
            "num_gaussians": num_gaussians, "delta_N": num_gaussians - self.last_n_gaussians,
            "peak_vram_GB": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
            "grad_cos_sim": float(grad_cos_sim), "parasitic_ratio": float(parasitic_ratio),
            "anchor_count": num_anchors,
            "tree_level_mean": float(self.gaussians.get_level.float().mean()),
            "spawn_ratio": float(num_gaussians / (num_anchors + 1e-7))
        }
        self.last_n_gaussians = num_gaussians
        
        histograms = {}
        if step % 1000 == 0:
            histograms["opacity"] = opacity.clone().detach() if opacity.shape[0] > 0 else torch.empty(0)
            if scaling.shape[0] > 0:
                histograms["scaling"] = scaling.clone().detach()
                scales_2d = scaling[:, :2] if scaling.shape[1] >= 2 else scaling
                gamma = scales_2d.max(dim=-1)[0] / (scales_2d.min(dim=-1)[0] + 1e-7)
                histograms["anisotropy"] = gamma.clone().detach()
            histograms["sh_dc_mag"] = self.gaussians.get_anchor_feat.detach().norm(dim=-1)

        return metrics, histograms
        
    def render(self, camera):
        was_training = self.gaussians.get_color_mlp.training
        self.gaussians.eval()
        try:
            with torch.no_grad():
                self.gaussians.set_anchor_mask(camera.camera_center, self.last_step, camera.resolution_scale)
                mask = prefilter_voxel(camera, self.gaussians, self.pipe, self.background)
                result = native_render(camera, self.gaussians, self.pipe, self.background, visible_mask=mask)
            return {"image": result["render"], "depth": result.get("depth")}
        finally:
            if was_training:
                self.gaussians.train()

    def save(self, save_dir, step):
        if os.path.realpath(save_dir) != os.path.realpath(self.scene.model_path):
            raise ValueError("save_dir must match the model_path used to construct Scene")
        self.scene.save(step)
        with open(os.path.join(save_dir, "anchor_gs_config.json"), "w") as config_file:
            json.dump({"method": 'octree_gs', "reference_preset": self.reference_preset,
                       "args": vars(self.args)}, config_file, indent=2)

    def load(self, model_path, iteration):
        checkpoint = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
        self.gaussians.load_ply_sparse_gaussian(os.path.join(checkpoint, "point_cloud.ply"))
        self.gaussians.load_mlp_checkpoints(checkpoint)
        self.gaussians.eval()
        self.last_step = iteration

    def get_spatial_centers(self):
        return self.gaussians.get_anchor

    def compute_physical_metrics(self, cameras=None):
        metrics = {}
        with torch.no_grad():
            if cameras is not None and len(cameras) > 0:
                cam = cameras[0]
                self.gaussians.set_anchor_mask(cam.camera_center, self.opt.iterations, cam.resolution_scale)
                voxel_visible_mask = prefilter_voxel(cam, self.gaussians, self.pipe, self.background)
                render_pkg = native_render(cam, self.gaussians, self.pipe, self.background, visible_mask=voxel_visible_mask)
                
                scales = render_pkg.get("scaling", torch.empty(0))
                opacities = render_pkg.get("neural_opacity", torch.empty(0))
                
                if scales.shape[0] > 0:
                    scales_2d = scales[:, :2] if scales.dim() > 1 and scales.shape[1] >= 2 else scales.unsqueeze(-1).expand(-1, 2)
                    max_S, _ = torch.max(scales_2d, dim=1)
                    min_S, _ = torch.min(scales_2d, dim=1)
                    gamma = max_S / (min_S + 1e-7)
                    
                    metrics["gamma_median"] = float(torch.median(gamma))
                    metrics["gamma_90th_percentile"] = float(torch.quantile(gamma, 0.90))
                    metrics["scale_mean"] = float(torch.mean(scales_2d))
                
                if opacities.shape[0] > 0:
                    metrics["alpha_mean"] = float(torch.mean(opacities))
                    
                view_dirs = []
                for c in cameras:
                    view_dirs.append(c.world_view_transform[:3, 2].tolist())
                view_dirs = F.normalize(torch.tensor(view_dirs, dtype=torch.float32, device="cuda"), dim=1)

                rots_tensor = render_pkg.get("rotations", torch.empty(0))
                if rots_tensor.shape[0] > 0:
                    rots = F.normalize(rots_tensor.clone(), dim=1)
                    w, x, y, z = rots.unbind(dim=-1)
                    normals = F.normalize(torch.stack([2*(x*z + w*y), 2*(y*z - w*x), 1-2*(x*x + y*y)], dim=-1), dim=1)
                    max_cos, _ = torch.max(torch.abs(torch.matmul(normals, view_dirs.T)), dim=1)
                    metrics["billboard_bias_ratio"] = float((max_cos > 0.90).float().mean())

        return metrics

    def evaluate_spatial_field(self, query_points: torch.Tensor, cameras=None) -> torch.Tensor:
        with torch.no_grad():
            V = query_points.shape[0]
            densities = torch.zeros(V, device="cuda")
            xyz = self.gaussians.get_anchor
            
            if xyz.shape[0] == 0:
                return densities

            chunk_size = 500_000
            for i in range(0, V, chunk_size):
                end = min(i + chunk_size, V)
                dist_sq = torch.cdist(query_points[i:end], xyz, p=2).pow(2)
                min_dist_sq, min_indices = torch.min(dist_sq, dim=1)
                densities[i:end] = torch.exp(-0.5 * min_dist_sq)
            return densities
