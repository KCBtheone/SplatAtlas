from pathlib import Path
import os
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)

import argparse
import shutil
import numpy as np
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.registry import METHOD_REGISTRY
import methods
from photometric import compute_photometric_metrics
from geometric import compute_geometric_pathology
from alignment import assess_manifold_collapse

def load_cameras_from_json(cam_path):
    if not os.path.exists(cam_path): return []
    with open(cam_path, 'r') as f: cams_data = json.load(f)
    class DummyCam:
        def __init__(self, center):
            self.camera_center = torch.tensor(center, dtype=torch.float32)
    return [DummyCam(c['center']) for c in cams_data if 'center' in c]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--render_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--ply_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--prior_type", type=str, default="none")
    parser.add_argument("--depth_dir", type=str, default="")
    parser.add_argument("--skip_quarantine", action="store_true")
    parser.add_argument("--colmap_dir", type=str, default="")
    return parser.parse_args()

def main():
    args = parse_args()
    Path(args.output_json).expanduser().parent.mkdir(parents=True, exist_ok=True)

    ply_path = Path(args.ply_path).expanduser().resolve()
    iteration_dir = ply_path.parent
    if not iteration_dir.name.startswith("iteration_") or iteration_dir.parent.name != "point_cloud":
        raise ValueError("Expected a PLY path under point_cloud/iteration_<step>/")
    iteration = int(iteration_dir.name.removeprefix("iteration_"))
    model_path = str(iteration_dir.parent.parent)

    model_class = methods.load_method(args.method)
    _OUTDOOR_360 = {"bicycle", "flowers", "garden", "stump", "treehill"}
    _res = 4 if args.scene in _OUTDOOR_360 else 2
    dataset_config = {"source_path": args.colmap_dir, "model_path": model_path, "resolution": _res}
    model = model_class(dataset_config, hyperparams={})

    try:
        model.load(model_path, iteration)
    except Exception as e:
        print(f"[Eval] 模型资产加载失败: {e}")
        return

    cams = load_cameras_from_json(os.path.join(model_path, "cameras.json"))

    geo_metrics = compute_geometric_pathology(model, cameras=cams)
    flags = geo_metrics.get("pathology_flags", [])

    colmap_ply_path = os.path.join(args.colmap_dir, "sparse", "0", "points3D.ply") if args.colmap_dir else None
    alignment_res = assess_manifold_collapse(model, model_path, colmap_ply_path=colmap_ply_path)

    photo_metrics = {}
    if alignment_res.get("manifold_collapse", False):
        flags.append("MANIFOLD_COLLAPSE")
        pass
    else:
        del model; torch.cuda.empty_cache()
        photo_metrics = compute_photometric_metrics(args.render_dir, args.gt_dir)

    if args.depth_dir and os.path.exists(args.depth_dir):
        depth_vars = [np.var(np.load(os.path.join(args.depth_dir, f))[np.load(os.path.join(args.depth_dir, f)) > 0.05])
                      for f in os.listdir(args.depth_dir) if f.endswith('.npy')]
        mean_depth_var = np.mean(depth_vars) if depth_vars else 1.0
        geo_metrics["mean_depth_variance"] = round(float(mean_depth_var), 4)
        if mean_depth_var < 0.01:
            flags.append("FLAT_WALL_COLLAPSE")

    geo_metrics["pathology_flags"] = flags

    report = {
        "method": args.method,
        "scene": args.scene,
        "prior_type": args.prior_type,
        "photometric": photo_metrics,
        "geometric": geo_metrics,
        "pathology_flags": list(set(flags))
    }

    with open(args.output_json, 'w') as f:
        json.dump(report, f, indent=4)

    if len(flags) > 0 and not args.skip_quarantine:
        master_log_path = os.path.join(os.path.dirname(model_path), "pathology_master_log.txt")
        with open(master_log_path, "a") as log_f:
            log_f.write(f"Method: {args.method} | Scene: {args.scene} | Path: {model_path} | Flags: {','.join(list(set(flags)))}\n")

if __name__ == "__main__":
    main()
