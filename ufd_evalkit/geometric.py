import os
import json
import numpy as np
import torch

def compute_geometric_pathology(model, cameras=None) -> dict:
    print(f"[UFD-EvalKit] Computing physical metrics")
    
    try:
        metrics = model.compute_physical_metrics(cameras=cameras)
    except NotImplementedError:
        print("  [UFD-EvalKit] Wrapper 未实现 compute_physical_metrics，返回空字典。")
        metrics = {}
    except Exception as e:
        return {"error": f"提取物理指标失败: {str(e)}"}

    flags = []
    if metrics.get("gamma_median", 0) > 18.0:
        flags.append("SEVERE_ANISOTROPY_DISTORTION")
    if metrics.get("alpha_mean", 1.0) < 0.05:
        flags.append("OPACITY_MINIMIZATION_COLLAPSE")
    if metrics.get("scale_mean", 0) > 5.0:
        flags.append("GLOBAL_SCALE_INFLATION")

    metrics["pathology_flags"] = flags
    return metrics

def depth_to_normal(depth):
    if len(depth.shape) == 2: depth = depth.unsqueeze(0).unsqueeze(0)
    elif len(depth.shape) == 3: depth = depth.unsqueeze(0)
    
    dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
    dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
    
    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))
    
    normal = torch.cat([-dx, -dy, torch.ones_like(depth)], dim=1)
    normal = torch.nn.functional.normalize(normal, dim=1)
    
    return (normal + 1.0) / 2.0
