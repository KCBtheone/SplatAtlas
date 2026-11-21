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

sys.path.append(os.path.abspath(f"{_project_paths.UPSTREAM_ROOT}/pgsr_official"))
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight
from utils.graphics_utils import patch_offsets, patch_warp
from gaussian_renderer import render as native_render
from scene import Scene, GaussianModel
from scene.app_model import AppModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from train import gen_virtul_cam

@register_method("pgsr")
class PGSRWrapper(BaseMethod):
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

        self.app_model = AppModel()
        self.app_model.train()
        self.app_model.cuda()

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
        gt_image, gt_image_gray = viewpoint_cam.get_image()
        
        if step > 1000 and self.opt.exposure_compensation:
            self.gaussians.use_app = True

        bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background
        
        render_pkg = native_render(viewpoint_cam, self.gaussians, self.pipe, bg, app_model=self.app_model,
                            return_plane=step>self.opt.single_view_weight_from_iter, 
                            return_depth_normal=step>self.opt.single_view_weight_from_iter)
                            
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        viewspace_point_tensor_abs = render_pkg.get("viewspace_points_abs", viewspace_point_tensor)
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        
        ssim_loss_val = (1.0 - ssim(image, gt_image))
        if 'app_image' in render_pkg and ssim_loss_val < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)
            
        image_loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * ssim_loss_val
        loss = image_loss.clone()
        
        loss_target = image_loss.clone()
        loss_parasitic = torch.tensor(0.0, device="cuda")
        
        if visibility_filter.sum() > 0:
            scale = self.gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            scale_l = self.opt.scale_loss_weight * min_scale_loss.mean()
            loss += scale_l
            loss_parasitic += scale_l

        normal_loss_val = 0.0
        if step > self.opt.single_view_weight_from_iter:
            weight = self.opt.single_view_weight
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]
            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            if not self.opt.wo_image_weight:
                normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
            else:
                normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
            loss += normal_loss
            loss_parasitic += normal_loss
            normal_loss_val = float(normal_loss)

        geo_loss_val = 0.0
        ncc_loss_val = 0.0
        if step > self.opt.multi_view_weight_from_iter:
            nearest_id = getattr(viewpoint_cam, 'nearest_id', [])
            nearest_cam = None if len(nearest_id) == 0 else self.scene.getTrainCameras()[random.sample(nearest_id, 1)[0]]
            use_virtul_cam = False
            if self.opt.use_virtul_cam and (np.random.random() < self.opt.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=self.dataset.multi_view_max_dis, deg_noise=self.dataset.multi_view_max_angle)
                use_virtul_cam = True
                
            if nearest_cam is not None:
                try:
                    patch_size = self.opt.multi_view_patch_size
                    sample_num = self.opt.multi_view_sample_num
                    pixel_noise_th = self.opt.multi_view_pixel_noise_th
                    total_patch_size = (patch_size * 2 + 1) ** 2
                    ncc_weight = self.opt.multi_view_ncc_weight
                    geo_weight = self.opt.multi_view_geo_weight

                    H, W = render_pkg['plane_depth'].squeeze().shape
                    ix, iy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
                    pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

                    nearest_render_pkg = native_render(nearest_cam, self.gaussians, self.pipe, bg, app_model=self.app_model, return_plane=True, return_depth_normal=False)

                    pts = self.gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                    pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                    map_z, d_mask = self.gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)
                    
                    pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                    pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                    R = torch.tensor(nearest_cam.R).float().cuda()
                    T = torch.tensor(nearest_cam.T).float().cuda()
                    pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                    pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                    pts_projections = torch.stack([
                        pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                        pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy
                    ], -1).float()
                    
                    pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                    if not self.opt.wo_use_geo_occ_aware:
                        d_mask = d_mask & (pixel_noise < pixel_noise_th)
                        weights = (1.0 / torch.exp(pixel_noise)).detach()
                        weights[~d_mask] = 0
                    else:
                        weights = torch.ones_like(pixel_noise)
                        weights[~d_mask] = 0

                    if d_mask.sum() > 0:
                        geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                        loss += geo_loss
                        loss_parasitic += geo_loss
                        geo_loss_val = float(geo_loss)

                    if not use_virtul_cam and d_mask.any():
                        with torch.no_grad():
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace=False)
                                valid_indices = valid_indices[index]
                            weights = weights.reshape(-1)[valid_indices]
                            pixels = pixels.reshape(-1,2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()
                            H, W = gt_image_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                            ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                        ref_local_n = render_pkg["rendered_normal"].permute(1,2,0).reshape(-1,3)[valid_indices]
                        ref_local_d = render_pkg['rendered_distance'].squeeze().reshape(-1)[valid_indices]
                        
                        H_ref_to_neareast = ref_to_neareast_r[None] - torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                        
                        K_nearest = nearest_cam.get_k(nearest_cam.ncc_scale)
                        if isinstance(K_nearest, np.ndarray): K_nearest = torch.from_numpy(K_nearest)
                        K_nearest = K_nearest.to(torch.float32).cuda()
                        
                        K_viewpoint_inv = viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)
                        if isinstance(K_viewpoint_inv, np.ndarray): K_viewpoint_inv = torch.from_numpy(K_viewpoint_inv)
                        K_viewpoint_inv = K_viewpoint_inv.to(torch.float32).cuda()

                        H_ref_to_neareast = torch.matmul(K_nearest[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ K_viewpoint_inv

                        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                        _, nearest_image_gray = nearest_cam.get_image()
                        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

                        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * weights
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0:
                            ncc_loss = ncc_weight * ncc.mean()
                            loss += ncc_loss
                            loss_parasitic += ncc_loss
                            ncc_loss_val = float(ncc_loss)
                except Exception as exc:
                    raise RuntimeError(
                        f"PGSR multi-view loss failed at step {step} "
                        f"for {viewpoint_cam.image_name!r} against {nearest_cam.image_name!r}"
                    ) from exc

        grad_cos_sim = 0.0
        parasitic_ratio = 0.0

        if self.track_decoupling and step % 100 == 0:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            self.app_model.optimizer.zero_grad(set_to_none=True)
            loss_target.backward(retain_graph=True)
            grad_target = self.gaussians._xyz.grad.clone() if self.gaussians._xyz.grad is not None else torch.zeros_like(self.gaussians._xyz)
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            self.app_model.optimizer.zero_grad(set_to_none=True)
            loss_parasitic.backward(retain_graph=True)
            grad_parasitic = self.gaussians._xyz.grad.clone() if self.gaussians._xyz.grad is not None else torch.zeros_like(self.gaussians._xyz)
            
            valid_mask = (torch.norm(grad_target, dim=1) > 0) & (torch.norm(grad_parasitic, dim=1) > 0)
            if valid_mask.any():
                grad_cos_sim = float(F.cosine_similarity(grad_target[valid_mask], grad_parasitic[valid_mask], dim=1).mean())
            parasitic_ratio = float(torch.norm(grad_parasitic, dim=1).mean() / (torch.norm(grad_target, dim=1).mean() + 1e-7))
            
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            self.app_model.optimizer.zero_grad(set_to_none=True)
            loss.backward()
        else:
            loss.backward()

        with torch.no_grad():
            if step < self.opt.densify_until_iter:
                obs_mask = (render_pkg.get("out_observe", torch.ones_like(visibility_filter)) > 0) & visibility_filter
                self.gaussians.max_radii2D[obs_mask] = torch.max(self.gaussians.max_radii2D[obs_mask], radii[obs_mask])
                self.gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)
                
                if step > self.opt.densify_from_iter and step % self.opt.densification_interval == 0:
                    size_threshold = 20 if step > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, self.opt.densify_abs_grad_threshold, self.opt.opacity_cull_threshold, self.scene.cameras_extent, size_threshold)
                    
            if self.opt.use_multi_view_trim and step % 1000 == 0 and step < self.opt.densify_until_iter:
                observe_the = 2
                observe_cnt = torch.zeros_like(self.gaussians.get_opacity)
                for view in self.scene.getTrainCameras():
                    render_pkg_tmp = native_render(view, self.gaussians, self.pipe, bg, app_model=self.app_model, return_plane=False, return_depth_normal=False)
                    out_observe = render_pkg_tmp["out_observe"]
                    observe_cnt[out_observe > 0] += 1
                prune_mask = (observe_cnt < observe_the).squeeze()
                if prune_mask.sum() > 0:
                    self.gaussians.prune_points(prune_mask)

            if step < self.opt.densify_until_iter:
                if step % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and step == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            self.gaussians.optimizer.step()
            self.app_model.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            self.app_model.optimizer.zero_grad(set_to_none=True)

        num_gaussians = self.gaussians.get_xyz.shape[0]
        metrics = {
            "loss": float(loss), "loss_l1": float(loss_target), "loss_ssim": float(ssim_loss_val),
            "loss_normal": float(normal_loss_val), "loss_geo": float(geo_loss_val), "loss_ncc": float(ncc_loss_val),
            "num_gaussians": int(num_gaussians), "delta_N": int(num_gaussians - self.last_n_gaussians),
            "peak_vram_GB": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
            "grad_cos_sim": float(grad_cos_sim), "parasitic_ratio": float(parasitic_ratio)
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
            render_pkg = native_render(camera, self.gaussians, self.pipe, self.background, app_model=self.app_model, return_plane=False, return_depth_normal=False)
        return {"image": render_pkg.get("app_image", render_pkg["render"]), "depth": render_pkg.get("plane_depth", None)}

    def save(self, save_dir, step):
        self.scene.save(step)
        self.app_model.save_weights(save_dir, step)

    def load(self, model_path, iteration):
        self.gaussians.load_ply(os.path.join(model_path, 'point_cloud', f'iteration_{iteration}', 'point_cloud.ply'))
        self.app_model.load_weights(model_path, iteration)

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
