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

_UPSTREAM = f'{_project_paths.UPSTREAM_ROOT}/scaffold_gs_official'
sys.path.insert(0, _UPSTREAM)
import os as _os
for _sub in ('utils', 'scene', 'gaussian_renderer', 'arguments', 'lpipsPyTorch', 'experiments'):
    _p = _os.path.join(_UPSTREAM, _sub)
    if _os.path.isdir(_p):
        sys.path.insert(0, _p)
del _os, _p, _sub, _UPSTREAM
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render as native_render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams

@register_method("scaffoldgs")
class ScaffoldGSWrapper(BaseMethod):
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
            self.args.voxel_size = 0.001
            self.args.update_init_factor = 4 if synthetic else 16
            if any(x in dataset_parts for x in ("tnt", "tanksandtemples", "deepblending")):
                self.args.voxel_size = 0.01

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
            feat_dim=self.dataset.feat_dim, n_offsets=self.dataset.n_offsets,
            voxel_size=self.dataset.voxel_size, update_depth=self.dataset.update_depth,
            update_init_factor=self.dataset.update_init_factor,
            update_hierachy_factor=self.dataset.update_hierachy_factor,
            use_feat_bank=self.dataset.use_feat_bank, appearance_dim=self.dataset.appearance_dim,
            ratio=self.dataset.ratio, add_opacity_dist=self.dataset.add_opacity_dist,
            add_cov_dist=self.dataset.add_cov_dist, add_color_dist=self.dataset.add_color_dist,
        )
        self.scene = Scene(self.dataset, self.gaussians, shuffle=False)
        self.gaussians.training_setup(self.opt)
        self.last_step = self.opt.iterations

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.viewpoint_stack = self.scene.getTrainCameras().copy()
        
        self.last_n_anchors = self.gaussians.get_anchor.shape[0] if hasattr(self.gaussians, 'get_anchor') else 0

    def train_iteration(self, step):
        self.gaussians.train()
        self.gaussians.update_learning_rate(step)
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        viewpoint_cam = self.viewpoint_stack.pop(random.randint(0, len(self.viewpoint_stack) - 1))
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, self.gaussians, self.pipe, self.background)
        render_pkg = native_render(
            viewpoint_cam, self.gaussians, self.pipe, self.background,
            visible_mask=voxel_visible_mask, retain_grad=step < self.opt.update_until,
        )
        image = render_pkg["render"]
        viewspace_points = render_pkg["viewspace_points"]
        visible = render_pkg["visibility_filter"]
        selection = render_pkg["selection_mask"]
        scaling = render_pkg["scaling"]
        opacity = render_pkg["neural_opacity"]
        gt = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt)
        ssim_loss = 1.0 - ssim(image, gt)
        loss_target = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * ssim_loss
        loss_parasitic = 0.01 * scaling.prod(dim=1).mean()
        loss = loss_target + loss_parasitic
        grad_cos_sim = parasitic_ratio = 0.0
        target_tensor = self.gaussians.get_anchor
        if self.track_decoupling and step % 100 == 0:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_target.backward(retain_graph=True)
            grad_target = target_tensor.grad.clone() if target_tensor.grad is not None else torch.zeros_like(target_tensor)
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_parasitic.backward(retain_graph=True)
            grad_parasitic = target_tensor.grad.clone() if target_tensor.grad is not None else torch.zeros_like(target_tensor)
            valid = (grad_target.norm(dim=1) > 0) & (grad_parasitic.norm(dim=1) > 0)
            if valid.any():
                grad_cos_sim = float(F.cosine_similarity(grad_target[valid], grad_parasitic[valid], dim=1).mean())
            parasitic_ratio = float(grad_parasitic.norm(dim=1).mean() / (grad_target.norm(dim=1).mean() + 1e-7))
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            viewspace_points.grad = None
        loss.backward()
        with torch.no_grad():
            if self.opt.start_stat < step < self.opt.update_until:
                self.gaussians.training_statis(viewspace_points, opacity, visible, selection, voxel_visible_mask)
                if step > self.opt.update_from and step % self.opt.update_interval == 0:
                    self.gaussians.adjust_anchor(
                        check_interval=self.opt.update_interval,
                        success_threshold=self.opt.success_threshold,
                        grad_threshold=self.opt.densify_grad_threshold,
                        min_opacity=self.opt.min_opacity,
                    )
            elif step == self.opt.update_until:
                del self.gaussians.opacity_accum
                del self.gaussians.offset_gradient_accum
                del self.gaussians.offset_denom
                torch.cuda.empty_cache()
            if step < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
        self.last_step = step
        num_anchors = int(self.gaussians.get_anchor.shape[0])
        metrics = {
            "loss": float(loss.detach()), "loss_l1": float(Ll1.detach()),
            "loss_ssim": float(ssim_loss.detach()), "loss_scaling": float(loss_parasitic.detach()),
            "num_gaussians": int(visible.sum()), "anchor_count": num_anchors,
            "delta_N": num_anchors - self.last_n_anchors,
            "peak_vram_GB": torch.cuda.max_memory_allocated() / 1024 ** 3,
            "grad_cos_sim": grad_cos_sim, "parasitic_ratio": parasitic_ratio,
        }
        self.last_n_anchors = num_anchors
        histograms = {}
        if step % 1000 == 0:
            histograms["opacity"] = opacity.detach()
            histograms["scaling"] = scaling.detach()
            scales_2d = scaling.detach()[:, :2]
            histograms["anisotropy"] = scales_2d.max(dim=1)[0] / (scales_2d.min(dim=1)[0] + 1e-7)
        return metrics, histograms

    def render(self, camera):
        was_training = self.gaussians.get_color_mlp.training
        self.gaussians.eval()
        try:
            with torch.no_grad():
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
            json.dump({"method": 'scaffoldgs', "reference_preset": self.reference_preset,
                       "args": vars(self.args)}, config_file, indent=2)

    def load(self, model_path, iteration):
        checkpoint = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
        self.gaussians.load_ply_sparse_gaussian(os.path.join(checkpoint, "point_cloud.ply"))
        self.gaussians.load_mlp_checkpoints(checkpoint)
        self.gaussians.eval()
        self.last_step = iteration

    def get_spatial_centers(self):
        return self.gaussians.get_anchor if hasattr(self.gaussians, 'get_anchor') else self.gaussians._xyz

    def compute_physical_metrics(self, cameras=None):
        metrics = {}
        with torch.no_grad():
            if hasattr(self.gaussians, 'get_scaling'):
                raw_scales = self.gaussians.get_scaling
                scales = torch.exp(raw_scales)
                
                scales_2d = scales[:, :2] if scales.dim() > 1 and scales.shape[1] >= 2 else scales.unsqueeze(-1).expand(-1, 2)
                
                max_S, _ = torch.max(scales_2d, dim=1)
                min_S, _ = torch.min(scales_2d, dim=1)
                gamma = max_S / (min_S + 1e-7)
                
                metrics["gamma_median"] = float(torch.median(gamma))
                metrics["gamma_90th_percentile"] = float(torch.quantile(gamma, 0.90))
                metrics["scale_mean"] = float(torch.mean(scales_2d))
                
            if hasattr(self.gaussians, 'get_opacity'):
                metrics["alpha_mean"] = float(torch.mean(torch.sigmoid(self.gaussians.get_opacity)))
                
            if cameras is not None and len(cameras) > 0 and hasattr(self.gaussians, 'get_rotation'):
                view_dirs = []
                for c in cameras:
                    view_dirs.append(c.world_view_transform[:3, 2].tolist())
                view_dirs = F.normalize(torch.tensor(view_dirs, dtype=torch.float32, device="cuda"), dim=1)

                rots = F.normalize(self.gaussians.get_rotation.clone(), dim=1)
                w, x, y, z = rots.unbind(dim=-1)
                normals = F.normalize(torch.stack([2*(x*z + w*y), 2*(y*z - w*x), 1-2*(x*x + y*y)], dim=-1), dim=1)
                
                max_cos, _ = torch.max(torch.abs(torch.matmul(normals, view_dirs.T)), dim=1)
                metrics["billboard_bias_ratio"] = float((max_cos > 0.90).float().mean())

        return metrics

    def evaluate_spatial_field(self, query_points: torch.Tensor, cameras=None) -> torch.Tensor:
        with torch.no_grad():
            V = query_points.shape[0]
            densities = torch.zeros(V, device="cuda")
            
            if not hasattr(self.gaussians, 'get_anchor') or not hasattr(self.gaussians, 'get_scaling'):
                return densities
                
            xyz = self.gaussians.get_anchor
            opacities = torch.sigmoid(self.gaussians.get_opacity).squeeze()
            scales = torch.exp(self.gaussians.get_scaling)
            sigma_sq = (scales[:, :2].max(dim=1)[0].pow(2)) if scales.shape[1] >= 2 else scales.squeeze().pow(2)
            
            N_gaussians = xyz.shape[0]
            chunk_size = max(1, 30_000_000 // (N_gaussians + 1))
            for i in range(0, V, chunk_size):
                end = min(i + chunk_size, V)
                dist_sq = torch.cdist(query_points[i:end], xyz, p=2).pow(2)
                weights = torch.exp(-0.5 * dist_sq / (sigma_sq.unsqueeze(0) + 1e-7))
                densities[i:end] = torch.sum(weights * opacities.unsqueeze(0), dim=1)
            return densities

