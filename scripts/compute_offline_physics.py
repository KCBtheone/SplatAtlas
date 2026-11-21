import os
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)

import argparse
import torch
import numpy as np
from scipy.stats import entropy
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.registry import METHOD_REGISTRY
import methods

def compute_gini(array):
    array = array.flatten()
    array = array[array > 0]
    if len(array) == 0: return 0.0
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def generate_standard_grid(cameras, resolution=64):
    if not cameras:
        min_bounds = torch.tensor([-5.0, -5.0, -5.0])
        max_bounds = torch.tensor([5.0, 5.0, 5.0])
    else:
        cam_centers = torch.stack([c.camera_center for c in cameras])
        min_bounds, _ = torch.min(cam_centers, dim=0)
        max_bounds, _ = torch.max(cam_centers, dim=0)
        extent = max_bounds - min_bounds
        min_bounds -= extent * 0.1
        max_bounds += extent * 0.1
    
    X = torch.linspace(min_bounds[0], max_bounds[0], resolution)
    Y = torch.linspace(min_bounds[1], max_bounds[1], resolution)
    Z = torch.linspace(min_bounds[2], max_bounds[2], resolution)
    grid_x, grid_y, grid_z = torch.meshgrid(X, Y, Z, indexing='ij')
    
    query_points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)
    return query_points.cuda()

def analyze_physics(model, cameras):
    stats = {}
    
    try:
        specific_metrics = model.compute_physical_metrics(cameras=cameras)
        stats.update(specific_metrics)
    except NotImplementedError:
        print("  [Physics] Wrapper 未实现 compute_physical_metrics，跳过特异性指标。")
    except Exception as e:
        print(f"  [Physics] 特异性指标计算发生错误: {e}")
    
    try:
        query_points = generate_standard_grid(cameras, resolution=64)
        with torch.no_grad():
            scalar_field = model.evaluate_spatial_field(query_points, cameras=cameras)
        
        field_np = scalar_field.cpu().numpy()
        stats['spatial_gini'] = compute_gini(field_np)
        
        p = field_np / (field_np.sum() + 1e-7)
        stats['density_entropy'] = entropy(p[p > 0])
        stats["Field_Total_Volume"] = float(np.sum(field_np))
    except NotImplementedError:
        print("  [Physics] Wrapper 未实现 evaluate_spatial_field，跳过空间分布诊断。")
    except Exception as e:
        print(f"  [Physics] 空间标量场评估发生异常: {e}")
        
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()}

def load_cameras_from_json(cam_path):
    if not os.path.exists(cam_path): return []
    with open(cam_path, 'r') as f: cams_data = json.load(f)
    
    class DummyCam:
        def __init__(self, center):
            self.camera_center = torch.tensor(center, dtype=torch.float32)
            
    return [DummyCam(c['center']) for c in cams_data if 'center' in c]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, help="模型方法名称")
    parser.add_argument("--source_path", type=str, default="", help="原始数据集路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型输出目录")
    parser.add_argument("--iteration", type=int, default=30000)
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_path)
    if not os.path.exists(model_dir):
        print(f"[Physics] 跳过: 模型目录不存在 {model_dir}")
        return
        
    folder_name = os.path.basename(model_dir)
    print(f"[Physics] 正在精准解剖场景: {folder_name} ...")

    dataset_config = {"source_path": args.source_path, "model_path": args.model_path, "resolution": 1}
    try:
        model_class = methods.load_method(args.method)
        model = model_class(dataset_config, hyperparams={})
        model.load(args.model_path, args.iteration)
    except Exception as e:
        print(f"[Physics] 实例化或加载模型失败: {e}")
        return

    cameras = model.scene.getTrainCameras() if hasattr(model, 'scene') else []

    stats = analyze_physics(model, cameras)
    
    if stats:
        stats['experiment'] = folder_name
        with open(os.path.join(model_dir, f"offline_physics_{args.iteration}.json"), 'w') as f:
            json.dump(stats, f, indent=4, cls=NpEncoder)
        print(f"  [Physics] {folder_name} 分析完成！提取指标数: {len(stats)}")

if __name__ == "__main__":
    main()
