from core import paths as _project_paths

import os
import sys
import random
import torch
import numpy as np
import torch.nn.functional as F
from argparse import ArgumentParser
from plyfile import PlyData, PlyElement

from core.registry import register_method
from core.base_method import BaseMethod

sys.path.append(os.path.abspath(f"{_project_paths.UPSTREAM_ROOT}/ges_official"))
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import apply_dog_filter
from gaussian_renderer import render_laplacian
from scene import Scene, LaplacianModel
from arguments import ModelParams, PipelineParams, OptimizationParams

@register_method("ges")
class GESWrapper(BaseMethod):
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

        self.gaussians = LaplacianModel(self.dataset.sh_degree)
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
        
        render_pkg = render_laplacian(viewpoint_cam, self.gaussians, self.pipe, self.background)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        
        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        
        freq = (step / self.opt.iterations) * 100
        mask = apply_dog_filter(image.unsqueeze(0), freq=freq, scale_factor=self.opt.im_laplace_scale_factor).squeeze(0)
        mask_loss = l1_loss(image * mask, gt_image * mask)

        loss_target = (1.0 - self.opt.lambda_dssim - self.opt.lambda_im_laplace) * Ll1
        loss_parasitic = self.opt.lambda_dssim * (1.0 - ssim_value) + self.opt.lambda_im_laplace * mask_loss
        loss = loss_target + loss_parasitic

        grad_cos_sim = 0.0
        parasitic_ratio = 0.0

        if self.track_decoupling and step % 100 == 0:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_target.backward(retain_graph=True)
            grad_target = self.gaussians._xyz.grad.clone() if self.gaussians._xyz.grad is not None else torch.zeros_like(self.gaussians._xyz)
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_parasitic.backward(retain_graph=True)
            grad_parasitic = self.gaussians._xyz.grad.clone() if self.gaussians._xyz.grad is not None else torch.zeros_like(self.gaussians._xyz)
            
            state = self.gaussians.optimizer.state.get(self.gaussians._xyz, None)
            if state is not None and "exp_avg_sq" in state:
                v_t = state["exp_avg_sq"]
                lr_xyz = 0.0
                for param_group in self.gaussians.optimizer.param_groups:
                    if param_group["name"] == "xyz":
                        lr_xyz = param_group["lr"]
                        break
                        
                u_target = (lr_xyz / (torch.sqrt(v_t) + 1e-15)) * grad_target
                u_parasitic = (lr_xyz / (torch.sqrt(v_t) + 1e-15)) * grad_parasitic
                
                norm_target = torch.norm(u_target, dim=1)
                norm_parasitic = torch.norm(u_parasitic, dim=1)
                valid_mask = (norm_target > 0) & (norm_parasitic > 0)
                
                if valid_mask.any():
                    cos_sim = F.cosine_similarity(u_target[valid_mask], u_parasitic[valid_mask], dim=1)
                    grad_cos_sim = float(cos_sim.mean())
                    
                    r_group = norm_parasitic[valid_mask] / (norm_target[valid_mask] + norm_parasitic[valid_mask] + 1e-7)
                    ti = r_group * torch.clamp(-cos_sim, min=0.0)
                    parasitic_ratio = float(ti.mean())

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
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, self.opt.prune_opacity_threshold, self.scene.cameras_extent, size_threshold)
                    
                if step > self.opt.densify_from_iter and step % self.opt.shape_pruning_interval == 0:
                    self.gaussians.size_prune(self.opt.prune_shape_threshold)
                
                if step % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and step == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()
                    
                if step % self.opt.shape_reset_interval == 0:
                    self.gaussians.reset_shape()

            if step < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        num_gaussians = self.gaussians.get_xyz.shape[0]
        metrics = {
            "loss": float(loss), "loss_l1": float(loss_target), "loss_ssim": float(loss_parasitic),
            "num_gaussians": int(num_gaussians), "delta_N": int(num_gaussians - self.last_n_gaussians),
            "peak_vram_GB": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
            "grad_cos_sim": float(grad_cos_sim), "parasitic_ratio": float(parasitic_ratio),
            "average_shape": float(self.gaussians.get_shape.mean())
        }
        self.last_n_gaussians = num_gaussians
        
        histograms = {}
        if step % 1000 == 0:
            histograms["opacity"] = torch.sigmoid(self.gaussians._opacity).clone().detach()
            scales = (torch.exp(self.gaussians._scaling) * self.gaussians.get_shape).clone().detach()
            histograms["scaling"] = scales
            scales_2d = scales[:, :2] if scales.shape[1] >= 2 else scales
            gamma = scales_2d.max(dim=-1)[0] / (scales_2d.min(dim=-1)[0] + 1e-7)
            histograms["anisotropy"] = gamma
            histograms["sh_dc_mag"] = self.gaussians._features_dc.detach().norm(dim=-1)
            histograms["shape"] = self.gaussians.get_shape.clone().detach()

        return metrics, histograms
        
    def render(self, camera):
        with torch.no_grad():
            render_pkg = render_laplacian(camera, self.gaussians, self.pipe, self.background)
        return {"image": render_pkg["render"], "depth": render_pkg.get("depth", None)}

    def save(self, save_dir, step):
        ply_path = os.path.join(save_dir, "point_cloud", f"iteration_{step}", "point_cloud.ply")
        os.makedirs(os.path.dirname(ply_path), exist_ok=True)
        model = self.gaussians
        log_scales = model._scaling.detach()
        shape = model.get_shape.detach()
        effective_log_scales = log_scales + shape.clamp_min(torch.finfo(shape.dtype).tiny).log()
        xyz = model.get_xyz.detach().cpu().numpy()
        scale_fields = [f"ges_log_scale_{i}" for i in range(log_scales.shape[1])]
        fields = model.construct_list_of_attributes() + scale_fields + ["ges_shape"]
        vertices = np.empty(len(xyz), dtype=[(field, "f4") for field in fields])
        values = np.concatenate((
            xyz, np.zeros_like(xyz),
            model._features_dc.detach().transpose(1, 2).flatten(1).cpu().numpy(),
            model._features_rest.detach().transpose(1, 2).flatten(1).cpu().numpy(),
            model._opacity.detach().cpu().numpy(),
            effective_log_scales.cpu().numpy(),
            model._rotation.detach().cpu().numpy(),
            log_scales.cpu().numpy(),
            model._shape.detach().cpu().numpy(),
        ), axis=1)
        if len(fields) != values.shape[1]:
            raise ValueError("GES PLY field count does not match the vertex data")
        for field, column in zip(fields, values.T):
            vertices[field] = column
        settings = np.array(
            [(model.active_sh_degree, model.shape_strngth)],
            dtype=[("active_sh_degree", "i4"), ("shape_strength", "f8")],
        )
        PlyData([
            PlyElement.describe(vertices, "vertex"),
            PlyElement.describe(settings, "ges_settings"),
        ]).write(ply_path)

    def load(self, model_path, iteration):
        ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
        model = self.gaussians
        model.load_ply(ply_path)
        ply = PlyData.read(ply_path)
        vertices = ply["vertex"].data
        if "ges_shape" not in vertices.dtype.names:
            return
        settings = ply["ges_settings"].data
        raw_scales = np.column_stack([
            vertices[f"ges_log_scale_{i}"] for i in range(model._scaling.shape[1])
        ])
        with torch.no_grad():
            model._scaling.copy_(torch.from_numpy(raw_scales))
            model._shape.copy_(torch.from_numpy(vertices["ges_shape"].copy()).unsqueeze(1))
        model.active_sh_degree = int(settings["active_sh_degree"][0])
        model.shape_strngth = float(settings["shape_strength"][0])

    def get_spatial_centers(self):
        return self.gaussians._xyz

    def compute_physical_metrics(self, cameras=None):
        metrics = {}
        with torch.no_grad():
            scales = torch.exp(self.gaussians._scaling) * self.gaussians.get_shape
            scales_2d = scales[:, :2] if scales.dim() > 1 and scales.shape[1] >= 2 else scales.unsqueeze(-1).expand(-1, 2)
            max_S, _ = torch.max(scales_2d, dim=1)
            min_S, _ = torch.min(scales_2d, dim=1)
            gamma = max_S / (min_S + 1e-7)
            metrics["gamma_median"] = float(torch.median(gamma))
            metrics["gamma_90th_percentile"] = float(torch.quantile(gamma, 0.90))
            metrics["scale_mean"] = float(torch.mean(scales_2d))
            metrics["alpha_mean"] = float(torch.mean(torch.sigmoid(self.gaussians._opacity)))
            metrics["shape_mean"] = float(torch.mean(self.gaussians.get_shape))
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

    def evaluate_spatial_field(self, query_points, cameras=None):
        with torch.no_grad():
            V = query_points.shape[0]
            densities = torch.zeros(V, device="cuda")
            xyz, opacities = self.gaussians._xyz, torch.sigmoid(self.gaussians._opacity).squeeze()
            scales = torch.exp(self.gaussians._scaling) * self.gaussians.get_shape
            sigma_sq = (scales[:, :2].max(dim=1)[0].pow(2)) if scales.shape[1] >= 2 else scales.squeeze().pow(2)
            N_gaussians = xyz.shape[0]
            chunk_size = max(1, 30_000_000 // (N_gaussians + 1))
            for i in range(0, V, chunk_size):
                end = min(i + chunk_size, V)
                dist_sq = torch.cdist(query_points[i:end], xyz, p=2).pow(2)
                weights = torch.exp(-0.5 * dist_sq / (sigma_sq.unsqueeze(0) + 1e-7))
                densities[i:end] = torch.sum(weights * opacities.unsqueeze(0), dim=1)
            return densities
