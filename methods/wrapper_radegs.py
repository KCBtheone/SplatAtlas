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

target_path = os.path.abspath(f"{_project_paths.UPSTREAM_ROOT}/RaDe-GS_offy")
if not os.path.exists(target_path):
    raise FileNotFoundError(f"Target directory {target_path} does not exist.")

if target_path not in sys.path:
    sys.path.insert(0, target_path)

for key in list(sys.modules.keys()):
    if key.split('.')[0] in ['utils', 'scene', 'gaussian_renderer', 'arguments', 'warp_patch_ncc', 'fused_ssim', 'simple_knn']:
        del sys.modules[key]

from utils.loss_utils import ssim, L1_loss_appearance, PatchMatch
from gaussian_renderer import render as native_render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import depth_double_to_normal, depth_to_normal

@register_method("radegs")
class RaDeGSWrapper(BaseMethod):
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
        self.args.data_device = "cpu"
        self.track_decoupling = hyperparams.get("track_decoupling", False)

        self.dataset = self.lp.extract(self.args)
        self.opt = self.op.extract(self.args)
        self.pipe = self.pp.extract(self.args)

        self.gaussians = GaussianModel(self.dataset.sh_degree)
        self.gaussians.max_points = int(os.environ.get("SPLATATLAS_RADE_MAX_POINTS", "0"))
        if self.gaussians.max_points:
            print("[USER_POINT_CAP] max_points=%d" % self.gaussians.max_points)
        self.scene = Scene(self.dataset, self.gaussians)
        self.gaussians.training_setup(self.opt)

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.kernel_size = self.dataset.kernel_size

        self.trainCameras = self.scene.getTrainCameras().copy()
        if self.dataset.disable_filter3D:
            self.gaussians.reset_3D_filter()
        else:
            self.gaussians.compute_3D_filter(cameras=self.trainCameras)

        self.patchmatch = None
        if self.opt.lambda_multi_view_ncc > 0 or self.opt.lambda_multi_view_geo > 0:
            self.patchmatch = PatchMatch(
                self.opt.multi_view_patch_size,
                self.opt.multi_view_pixel_noise_th,
                kernel_size=self.kernel_size,
                pipe=self.pipe,
                debug=False,
                model_path=self.dataset.model_path,
            )

        self.viewpoint_stack = None
        self.last_n_gaussians = len(self.gaussians.get_xyz)

    def train_iteration(self, step):
        self.gaussians.update_learning_rate(step)
        if step % 1000 == 0:
            self.gaussians.oneupSHdegree()

        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()

        viewpoint_cam = self.viewpoint_stack.pop(random.randint(0, len(self.viewpoint_stack) - 1))
        reg_kick_on = step >= self.opt.regularization_from_iter

        render_pkg = native_render(
            viewpoint_cam,
            self.gaussians,
            self.pipe,
            self.background,
            self.kernel_size,
            require_depth=reg_kick_on
        )
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = L1_loss_appearance(image, gt_image, self.gaussians, viewpoint_cam.uid)
        ssim_value = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))

        loss_target = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim_value)

        depth_normal_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        ncc_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        geo_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        depth_normal = None

        if reg_kick_on and self.opt.lambda_depth_normal > 0:
            rendered_expected_depth = render_pkg["expected_depth"]
            rendered_median_depth = render_pkg["median_depth"]
            rendered_normal = render_pkg["normal"]
            if 0.0 < self.dataset.depth_ratio < 1.0:
                depth_normal = depth_double_to_normal(viewpoint_cam, rendered_expected_depth, rendered_median_depth)
                normal_error_map = 1 - torch.linalg.vecdot(rendered_normal.unsqueeze(0), depth_normal, dim=1)
                depth_normal_loss = (1 - self.dataset.depth_ratio) * normal_error_map[0].mean() + self.dataset.depth_ratio * normal_error_map[1].mean()
                depth_normal = None
            else:
                depth_map = rendered_expected_depth if self.dataset.depth_ratio < 1.0 else rendered_median_depth
                depth_normal = depth_to_normal(viewpoint_cam, depth_map)
                normal_error_map = 1 - torch.linalg.vecdot(rendered_normal, depth_normal, dim=0)
                depth_normal_loss = normal_error_map.mean()

        if reg_kick_on and self.opt.lambda_multi_view_ncc > 0 and self.patchmatch is not None:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else self.scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id, 1)[0]]
            ncc_loss, geo_loss = self.patchmatch(self.gaussians, render_pkg, viewpoint_cam, nearest_cam, step, depth_normal)

        loss_parasitic = self.opt.lambda_depth_normal * depth_normal_loss + self.opt.lambda_multi_view_ncc * ncc_loss + self.opt.lambda_multi_view_geo * geo_loss
        loss = loss_target + loss_parasitic

        grad_cos_sim = 0.0
        parasitic_ratio = 0.0
        sti_stats = {"sti_spatial": 0.0, "sti_geometry": 0.0, "sti_opacity": 0.0, "sti_appearance": 0.0}

        if self.track_decoupling and step % 100 == 0:
            param_groups_map = {
                "spatial": [self.gaussians._xyz],
                "geometry": [self.gaussians._scaling, self.gaussians._rotation],
                "opacity": [self.gaussians._opacity],
                "appearance": [self.gaussians._features_dc, self.gaussians._features_rest],
            }

            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss_target.backward(retain_graph=True)
            grad_t_global = []
            grads_target = {}
            for group_name, params in param_groups_map.items():
                group_grads = []
                for p in params:
                    if p.grad is not None:
                        group_grads.append(p.grad.clone().reshape(-1))
                        grad_t_global.append(p.grad.clone().reshape(-1))
                if group_grads:
                    grads_target[group_name] = torch.cat(group_grads)

            self.gaussians.optimizer.zero_grad(set_to_none=True)
            if hasattr(loss_parasitic, 'requires_grad') and loss_parasitic.requires_grad:
                loss_parasitic.backward(retain_graph=True)
            grad_p_global = []
            grads_parasitic = {}
            for group_name, params in param_groups_map.items():
                group_grads = []
                for p in params:
                    if p.grad is not None:
                        group_grads.append(p.grad.clone().reshape(-1))
                        grad_p_global.append(p.grad.clone().reshape(-1))
                if group_grads:
                    grads_parasitic[group_name] = torch.cat(group_grads)

            if grad_t_global and grad_p_global:
                vec_t = torch.cat(grad_t_global)
                vec_p = torch.cat(grad_p_global)
                norm_t = vec_t.norm()
                norm_p = vec_p.norm()
                if norm_t > 0 and norm_p > 0:
                    grad_cos_sim = float(F.cosine_similarity(vec_t.unsqueeze(0), vec_p.unsqueeze(0)))
                    parasitic_ratio = float(norm_p / (norm_t + norm_p + 1e-7))

            for group_name in param_groups_map:
                gt = grads_target.get(group_name)
                gp = grads_parasitic.get(group_name)
                if gt is not None and gp is not None:
                    lr_val = 0.001
                    for pg in self.gaussians.optimizer.param_groups:
                        if pg["name"] == group_name or (group_name == "spatial" and pg["name"] == "xyz") or (group_name == "geometry" and pg["name"] in ["scaling", "rotation"]):
                            lr_val = pg["lr"]
                            break
                    
                    eps = 1e-15
                    v_t_list = []
                    v_p_list = []
                    for p in param_groups_map[group_name]:
                        state = self.gaussians.optimizer.state.get(p, None)
                        if state is not None and "exp_avg_sq" in state:
                            v = state["exp_avg_sq"]
                            v_t_list.append(v.reshape(-1))
                            v_p_list.append(v.reshape(-1))
                        else:
                            v_t_list.append(torch.ones_like(p).reshape(-1))
                            v_p_list.append(torch.ones_like(p).reshape(-1))
                    
                    v_t_tensor = torch.cat(v_t_list)
                    v_p_tensor = torch.cat(v_p_list)

                    u_t = (lr_val / (torch.sqrt(v_t_tensor) + eps)) * gt
                    u_p = (lr_val / (torch.sqrt(v_p_tensor) + eps)) * gp
                    
                    norm_u_t = u_t.norm()
                    norm_u_p = u_p.norm()
                    
                    if norm_u_t > 0 and norm_u_p > 0:
                        cos_val = float(F.cosine_similarity(u_t.unsqueeze(0), u_p.unsqueeze(0)))
                        r_val = float(norm_u_p / (norm_u_t + norm_u_p + 1e-7))
                        ti = r_val * max(0.0, -cos_val)
                    else:
                        ti = 0.0
                    sti_stats[f"sti_{group_name}"] = ti

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
                    if self.dataset.disable_filter3D:
                        self.gaussians.reset_3D_filter()
                    else:
                        self.gaussians.compute_3D_filter(cameras=self.trainCameras)
                
                if step % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and step == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            if step % 100 == 0 and step > self.opt.densify_until_iter and not self.dataset.disable_filter3D:
                if step < self.opt.iterations - 100:
                    self.gaussians.compute_3D_filter(cameras=self.trainCameras)

            if step < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        num_gaussians = self.gaussians.get_xyz.shape[0]
        metrics = {
            "loss": float(loss),
            "loss_l1": float(loss_target),
            "loss_ssim": float(loss_parasitic),
            "num_gaussians": int(num_gaussians),
            "delta_N": int(num_gaussians - self.last_n_gaussians),
            "peak_vram_GB": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
            "grad_cos_sim": float(grad_cos_sim),
            "parasitic_ratio": float(parasitic_ratio),
            "ncc_loss": float(ncc_loss),
            "depth_normal_loss": float(depth_normal_loss)
        }
        metrics.update(sti_stats)
        self.last_n_gaussians = num_gaussians
        
        histograms = {}
        if step % 1000 == 0:
            histograms["opacity"] = torch.sigmoid(self.gaussians._opacity).clone().detach()
            scales = self.gaussians.get_scaling.clone().detach()
            histograms["scaling"] = scales
            scales_2d = scales[:, :2] if scales.shape[1] >= 2 else scales
            gamma = scales_2d.max(dim=-1)[0] / (scales_2d.min(dim=-1)[0] + 1e-7)
            histograms["anisotropy"] = gamma
            histograms["sh_dc_mag"] = self.gaussians._features_dc.detach().norm(dim=-1)

        return metrics, histograms
        
    def render(self, camera):
        with torch.no_grad():
            render_pkg = native_render(camera, self.gaussians, self.pipe, self.background, self.kernel_size)
        return {"image": render_pkg["render"], "depth": render_pkg.get("expected_depth", None)}

    def save(self, save_dir, step):
        self.scene.save(step)

    def load(self, model_path, iteration):
        self.gaussians.load_ply(os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))

    def get_spatial_centers(self):
        return self.gaussians._xyz

    def compute_physical_metrics(self, cameras=None):
        metrics = {}
        with torch.no_grad():
            scales = self.gaussians.get_scaling
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
