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

sys.path.append(os.path.abspath(f"{_project_paths.UPSTREAM_ROOT}/mini_splatting_official"))
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_imp, render_depth
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.sh_utils import SH2RGB

@register_method("minisplatting")
class MiniSplattingWrapper(BaseMethod):
    def __init__(self, dataset_config, hyperparams):
        self.parser = ArgumentParser()
        self.lp = ModelParams(self.parser)
        self.op = OptimizationParams(self.parser)
        self.pp = PipelineParams(self.parser)
        self.parser.add_argument("--simp_iteration1", type=int, default=15000)
        self.parser.add_argument("--simp_iteration2", type=int, default=20000)
        self.parser.add_argument("--num_depth", type=int, default=3500000)
        self.parser.add_argument("--num_max", type=int, default=4500000)
        self.parser.add_argument("--sampling_factor", type=float, default=0.5)
        self.parser.add_argument("--imp_metric", type=str, default=None)
        
        self.args, _ = self.parser.parse_known_args([])
        
        self.args.source_path = dataset_config["source_path"]
        self.args.model_path = dataset_config["model_path"]
        self.args.eval = True
        self.args.resolution = dataset_config.get("resolution", 1)
        self.track_decoupling = hyperparams.get("track_decoupling", False)

        path_lower = str(self.args.source_path).lower()
        if "360" in path_lower and not any(x in path_lower for x in ["room", "counter", "kitchen", "bonsai"]):
            self.args.imp_metric = "outdoor"
        else:
            self.args.imp_metric = "indoor"

        self.dataset = self.lp.extract(self.args)
        self.opt = self.op.extract(self.args)
        self.pipe = self.pp.extract(self.args)

        self.gaussians = GaussianModel(self.dataset.sh_degree)
        self.scene = Scene(self.dataset, self.gaussians)
        self.gaussians.training_setup(self.opt)

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.viewpoint_stack = self.scene.getTrainCameras().copy()
        
        self.last_n_gaussians = len(self.gaussians.get_xyz)
        self.mask_blur = torch.zeros(self.last_n_gaussians, device="cuda", dtype=torch.bool)

    def train_iteration(self, step):
        if step < self.args.simp_iteration1:
            self.gaussians.update_learning_rate(step)
        else:
            self.gaussians.update_learning_rate(step - self.args.simp_iteration1 + 5000)

        if step % 1000 == 0 and step > self.args.simp_iteration1:
            self.gaussians.oneupSHdegree()
            
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
            
        viewpoint_cam = self.viewpoint_stack.pop(random.randint(0, len(self.viewpoint_stack) - 1))
        
        render_pkg = render_imp(viewpoint_cam, self.gaussians, self.pipe, self.background)
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

        if self.track_decoupling and step % 100 == 0:
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
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            loss.backward()
        else:
            loss.backward()

        is_reinit_step = False
        is_simplification_step = False
        blur_mask_active_ratio = 0.0

        with torch.no_grad():
            if step < self.opt.densify_until_iter:
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                area_max = render_pkg.get("area_max", torch.zeros_like(visibility_filter, dtype=torch.float32))
                if area_max.shape[0] == self.gaussians._xyz.shape[0]:
                    condition = area_max > (image.shape[1] * image.shape[2] / 5000.0)
                    if self.mask_blur.shape[0] != self.gaussians._xyz.shape[0]:
                        self.mask_blur = torch.zeros(self.gaussians._xyz.shape[0], device="cuda", dtype=torch.bool)
                    self.mask_blur = torch.logical_or(self.mask_blur, condition)
                
                if self.mask_blur.shape[0] > 0:
                    blur_mask_active_ratio = float(self.mask_blur.float().mean())

                if step > self.opt.densify_from_iter and step % self.opt.densification_interval == 0 and step % 5000 != 0 and self.gaussians._xyz.shape[0] < self.args.num_max:
                    size_threshold = 20 if step > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune_split(self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold, self.mask_blur)
                    self.mask_blur = torch.zeros(self.gaussians._xyz.shape[0], device="cuda", dtype=torch.bool)
                
                if step % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and step == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

                if step % 5000 == 0:
                    is_reinit_step = True
                    out_pts_list = []
                    gt_list = []
                    views = self.scene.getTrainCameras()
                    for view in views:
                        gt = view.original_image[0:3, :, :]
                        render_depth_pkg = render_depth(view, self.gaussians, self.pipe, self.background)
                        out_pts = render_depth_pkg["out_pts"]
                        accum_alpha = render_depth_pkg["accum_alpha"]
                        prob = 1.0 - accum_alpha
                        prob_sum = prob.sum()
                        if prob_sum > 0:
                            prob = prob / prob_sum
                        prob = prob.reshape(-1).cpu().numpy()
                        factor = 1.0 / (image.shape[1] * image.shape[2] * len(views) / self.args.num_depth)
                        N_xyz = prob.shape[0]
                        num_sampled = int(N_xyz * factor)
                        if num_sampled > 0 and num_sampled <= N_xyz:
                            indices = np.random.choice(N_xyz, size=num_sampled, p=prob, replace=False)
                            out_pts = out_pts.permute(1,2,0).reshape(-1,3)
                            gt = gt.permute(1,2,0).reshape(-1,3)
                            out_pts_list.append(out_pts[indices])
                            gt_list.append(gt[indices])
                    if out_pts_list:
                        out_pts_merged = torch.cat(out_pts_list)
                        gt_merged = torch.cat(gt_list)
                        self.gaussians.reinitial_pts(out_pts_merged, gt_merged)
                        self.gaussians.training_setup(self.opt)
                        self.mask_blur = torch.zeros(self.gaussians._xyz.shape[0], device="cuda", dtype=torch.bool)
                        torch.cuda.empty_cache()
                        self.viewpoint_stack = self.scene.getTrainCameras().copy()

            if step == self.args.simp_iteration1:
                is_simplification_step = True
                imp_score = torch.zeros(self.gaussians._xyz.shape[0], device="cuda")
                accum_area_max = torch.zeros(self.gaussians._xyz.shape[0], device="cuda")
                views = self.scene.getTrainCameras()
                for view in views:
                    r_pkg = render_imp(view, self.gaussians, self.pipe, self.background)
                    accum_weights = r_pkg["accum_weights"]
                    area_proj = r_pkg["area_proj"]
                    area_max = r_pkg["area_max"]
                    accum_area_max = accum_area_max + area_max
                    if self.args.imp_metric == 'outdoor':
                        mask_t = area_max != 0
                        temp = imp_score + accum_weights / (area_proj + 1e-7)
                        imp_score[mask_t] = temp[mask_t]
                    else:
                        imp_score = imp_score + accum_weights
                imp_score[accum_area_max == 0] = 0
                imp_sum = imp_score.sum()
                if imp_sum > 0:
                    prob = imp_score / imp_sum
                else:
                    prob = imp_score
                prob = prob.cpu().numpy()
                factor = self.args.sampling_factor
                N_xyz = self.gaussians._xyz.shape[0]
                num_sampled = int(N_xyz * factor * ((prob != 0).sum() / prob.shape[0]))
                if num_sampled > 0 and num_sampled <= N_xyz:
                    indices = np.random.choice(N_xyz, size=num_sampled, p=prob, replace=False)
                    mask = np.zeros(N_xyz, dtype=bool)
                    mask[indices] = True
                    self.gaussians.prune_points(~torch.tensor(mask, device="cuda"))
                    self.gaussians.max_sh_degree = self.dataset.sh_degree
                    self.gaussians.reinitial_pts(self.gaussians._xyz, SH2RGB(self.gaussians._features_dc + 0.0)[:,0])
                    self.gaussians.training_setup(self.opt)
                    self.mask_blur = torch.zeros(self.gaussians._xyz.shape[0], device="cuda", dtype=torch.bool)
                    torch.cuda.empty_cache()
                    self.viewpoint_stack = self.scene.getTrainCameras().copy()

            if step == self.args.simp_iteration2:
                is_simplification_step = True
                imp_score = torch.zeros(self.gaussians._xyz.shape[0], device="cuda")
                accum_area_max = torch.zeros(self.gaussians._xyz.shape[0], device="cuda")
                views = self.scene.getTrainCameras()
                for view in views:
                    r_pkg = render_imp(view, self.gaussians, self.pipe, self.background)
                    accum_weights = r_pkg["accum_weights"]
                    area_proj = r_pkg["area_proj"]
                    area_max = r_pkg["area_max"]
                    accum_area_max = accum_area_max + area_max
                    if self.args.imp_metric == 'outdoor':
                        mask_t = area_max != 0
                        temp = imp_score + accum_weights / (area_proj + 1e-7)
                        imp_score[mask_t] = temp[mask_t]
                    else:
                        imp_score = imp_score + accum_weights
                imp_score[accum_area_max == 0] = 0
                
                vals, idx = torch.sort(imp_score + 1e-6)
                cumsum_val = torch.cumsum(vals, dim=0)
                split_indices = ((cumsum_val / vals.sum()) > (1.0 - 0.99)).nonzero()
                if len(split_indices) > 0:
                    split_index = split_indices.min()
                    split_val_nonprune = vals[split_index]
                    non_prune_mask = imp_score > split_val_nonprune
                else:
                    non_prune_mask = torch.ones_like(imp_score).bool()
                    
                self.gaussians.prune_points(~non_prune_mask)
                self.gaussians.training_setup(self.opt)
                self.mask_blur = torch.zeros(self.gaussians._xyz.shape[0], device="cuda", dtype=torch.bool)
                torch.cuda.empty_cache()

            if step < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        num_gaussians = self.gaussians.get_xyz.shape[0]
        metrics = {
            "loss": float(loss), "loss_l1": float(loss_target), "loss_ssim": float(loss_parasitic),
            "num_gaussians": int(num_gaussians), "delta_N": int(num_gaussians - self.last_n_gaussians),
            "peak_vram_GB": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
            "grad_cos_sim": float(grad_cos_sim), "parasitic_ratio": float(parasitic_ratio),
            "blur_mask_active_ratio": float(blur_mask_active_ratio),
            "is_reinit_step": int(is_reinit_step),
            "is_simplification_step": int(is_simplification_step)
        }
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
            render_pkg = render_imp(camera, self.gaussians, self.pipe, self.background)
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
