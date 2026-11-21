if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import os
import json
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys
import time
import concurrent.futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ufd_evalkit.geometric import depth_to_normal
import methods

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=methods.SUPPORTED_METHODS, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5000, 10000, 20000, 30000])
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--images", type=str, default="images")
    parser.add_argument("--track_decoupling", action="store_true")
    parser.add_argument("--cap_gaussians", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--background", choices=["black", "white"], default=None)
    parser.add_argument("--skip_render", action="store_true")
    return parser.parse_args()

def save_np(data, path): np.save(path, data)

def in_memory_render(model, model_path, iteration):
    print("\n[Memory Fusion] Flushing VRAM fragments...")
    torch.cuda.empty_cache()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
    futures = []

    with torch.no_grad():
        test_render_dir = os.path.join(model_path, f"renders_test_{iteration}")
        test_depth_dir = os.path.join(model_path, f"depths_test_{iteration}")
        test_normal_dir = os.path.join(model_path, f"normals_test_{iteration}")
        test_gt_dir = os.path.join(model_path, f"gt_test_{iteration}")
        for d in [test_render_dir, test_depth_dir, test_normal_dir, test_gt_dir]: os.makedirs(d, exist_ok=True)

        print(f"Launching async full-dimensional rendering | Test Set")
        for idx, view in enumerate(tqdm(model.scene.getTestCameras(), desc="Rendering Test")):
            render_pkg = model.render(view)
            rendering = render_pkg.get("render", render_pkg.get("image")).detach().cpu()
            depth = render_pkg.get("depth", None)

            futures.append(executor.submit(torchvision.utils.save_image, rendering, os.path.join(test_render_dir, f'{idx:05d}.png')))
            futures.append(executor.submit(torchvision.utils.save_image, view.original_image.cpu(), os.path.join(test_gt_dir, f'{idx:05d}.png')))

            if depth is not None:
                depth_cpu = depth.detach().cpu()
                normal = render_pkg.get("normal", depth_to_normal(view, depth)).detach().cpu()
                futures.append(executor.submit(save_np, depth_cpu.numpy(), os.path.join(test_depth_dir, f'{idx:05d}.npy')))
                futures.append(executor.submit(torchvision.utils.save_image, normal, os.path.join(test_normal_dir, f'{idx:05d}.png')))

        train_render_dir = os.path.join(model_path, f"renders_train_{iteration}")
        train_depth_dir = os.path.join(model_path, f"depths_train_{iteration}")
        for d in [train_render_dir, train_depth_dir]: os.makedirs(d, exist_ok=True)

        print(f"Launching async full-dimensional rendering | Train Set")
        for idx, view in enumerate(tqdm(model.scene.getTrainCameras(), desc="Rendering Train")):
            render_pkg = model.render(view)
            rendering = render_pkg.get("render", render_pkg.get("image")).detach().cpu()
            futures.append(executor.submit(torchvision.utils.save_image, rendering, os.path.join(train_render_dir, f'{idx:05d}.png')))

            depth = render_pkg.get("depth", None)
            if depth is not None:
                futures.append(executor.submit(save_np, depth.detach().cpu().numpy(), os.path.join(train_depth_dir, f'{idx:05d}.npy')))

    print("[Async I/O] GPU rendering complete. Waiting for CPU flush...")
    concurrent.futures.wait(futures)

    error_log_path = os.path.join(model_path, "render_error.log")
    has_error = False
    for future in futures:
        try:
            future.result()
        except Exception as e:
            if not has_error:
                print(f"\n[CRITICAL ERROR] Async render thread crashed. See {error_log_path}")
                has_error = True
            with open(error_log_path, "a") as err_f:
                err_f.write(f"Exception: {str(e)}\n")

    executor.shutdown()

    if not has_error:
        flag_path = os.path.join(model_path, f"render_complete_{iteration}.flag")
        with open(flag_path, "w") as f: f.write("Completed by Async Engine")
        print("Flush complete. Flag written.")
    else:
        print("[WARNING] Render flag not written due to errors.")


def record_training_runtime(writer, model, step, start_time, stats):
    writer.add_scalar("runtime/elapsed_wall_seconds_current_process", time.time() - start_time, step)
    if torch.cuda.is_available():
        writer.add_scalar("runtime/cuda_allocated_bytes", torch.cuda.memory_allocated(), step)
        writer.add_scalar("runtime/cuda_reserved_bytes", torch.cuda.memory_reserved(), step)
        writer.add_scalar("runtime/cuda_peak_allocated_bytes_process", torch.cuda.max_memory_allocated(), step)
    seen = set()
    for owner_name, owner in (("model", model), ("gaussians", getattr(model, "gaussians", None))):
        optimizer = getattr(owner, "optimizer", None) if owner is not None else None
        if not isinstance(optimizer, torch.optim.Optimizer) or id(optimizer) in seen:
            continue
        seen.add(id(optimizer))
        for i, group in enumerate(optimizer.param_groups):
            lr = group.get("lr")
            if lr is not None:
                name = str(group.get("name", "group")).replace("/", "_").replace(" ", "_")
                writer.add_scalar(f"learning_rate/{owner_name}/{i}_{name}", float(lr), step)


def main():
    import random
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.model_path, exist_ok=True)
    tb_writer = SummaryWriter(args.model_path)

    dataset_config = {"source_path": args.source_path, "model_path": args.model_path, "resolution": args.resolution, "images": args.images}
    if args.background is not None:
        dataset_config["white_background"] = args.background == "white"
    method_hyperparams = {"track_decoupling": args.track_decoupling, "cap_gaussians": args.cap_gaussians}
    if args.method in ("scaffoldgs", "octree_gs"):
        method_hyperparams["iterations"] = args.iterations
    model = methods.load_method(args.method)(dataset_config, method_hyperparams)
    observed_config = {"cli": vars(args), "dataset_config": dataset_config, "method_hyperparams": method_hyperparams,
                       "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else None}
    for config_name in ("args", "opt"):
        config = getattr(model, config_name, None)
        if config is not None and hasattr(config, "__dict__"):
            observed_config["wrapper_" + config_name] = vars(config)
    tb_writer.add_text("run/observed_configuration", json.dumps(observed_config, default=str, indent=2), 0)
    print(f"[SplatAtlas] Starting training | Method: {args.method} | Resolution: {args.resolution}")

    num_train = len(model.scene.getTrainCameras())
    num_test = len(model.scene.getTestCameras()) if hasattr(model.scene, "getTestCameras") else 0
    split_info = (
        f"========================================\n"
        f"[Dataset Topology Checker]\n"
        f"Method: {args.method}\n"
        f"Train Cameras: {num_train}\n"
        f"Test Cameras:  {num_test}\n"
        f"Test split present: {num_test > 0}\n"
        f"========================================"
    )
    print(split_info)

    with open(os.path.join(args.model_path, "dataset_split.log"), "w") as f:
        f.write(split_info + "\n")

    cam_json_path = os.path.join(args.model_path, "training_camera_monitor.json" if args.method == "gslpm" else "cameras.json")
    with open(cam_json_path, "w") as f:
        json.dump([{"id": i, "center": c.camera_center.detach().cpu().numpy().tolist(), "view_dir": c.world_view_transform[:3, 2].detach().cpu().numpy().tolist()} for i, c in enumerate([c for item in (model.scene.getTrainCameras().values() if isinstance(model.scene.getTrainCameras(), dict) else model.scene.getTrainCameras()) for c in (item if isinstance(item, list) else [item])])], f, indent=4)

    t_train_start = time.time()
    timing_path = os.path.join(args.model_path, "timing.json")
    with open(timing_path, "w") as f:
        json.dump({"train_start_unix": t_train_start, "status": "running"}, f)

    cumulative_gpu_time = 0.0
    for step in range(1, args.iterations + 1):
        t_step_start = time.time()

        stats_or_tuple = model.train_iteration(step)

        if isinstance(stats_or_tuple, tuple):
            stats, histograms = stats_or_tuple
        else:
            stats, histograms = stats_or_tuple, {}

        step_elapsed = time.time() - t_step_start
        cumulative_gpu_time += step_elapsed
        stats['cumulative_gpu_time_sec'] = cumulative_gpu_time
        stats['step_time_ms'] = step_elapsed * 1000.0
        if step % 100 == 0:
            print(f"[TIMING] step={step} step_time_ms={stats.get('step_time_ms',0):.2f} "
                  f"iter_time_ms={stats.get('iter_time_ms',0):.2f} "
                  f"peak_vram_GB={stats.get('peak_vram_GB',0):.3f}", flush=True)

        if stats.get("delta_N") is not None and stats.get("delta_N") != 0:
            tb_writer.add_scalar("events/wrapper_reported_net_primitive_change", stats["delta_N"], step)
        if step % 100 == 0:
            record_training_runtime(tb_writer, model, step, t_train_start, stats)
            print(f"\n{'='*40}")
            print(f"Step {step:05d} Runtime Monitor")
            print(f"{'='*40}")

            print("[Scalars]")
            for key, val in stats.items():
                if val is not None:
                    tb_writer.add_scalar(f"train/{key}", val, step)
                    if isinstance(val, float):
                        print(f"  |- {key}: {val:.6f}")
                    else:
                        print(f"  |- {key}: {val}")

        if histograms:
            print("\n[Histograms]")
            for dist_name, tensor_data in histograms.items():
                if not torch.isnan(tensor_data).any():
                    try:
                        arr = tensor_data.float().cpu().numpy().flatten()
                        arr = arr[np.isfinite(arr)]
                        if len(arr) > 0:
                            counts, bin_edges = np.histogram(arr.astype(np.float64), bins=100)
                            tb_writer.add_histogram_raw(
                                f'Distributions/{dist_name}',
                                min=float(arr.min()), max=float(arr.max()),
                                num=len(arr),
                                sum=float(arr.sum()),
                                sum_squares=float((arr.astype(np.float64)**2).sum()),
                                bucket_limits=bin_edges[1:].tolist(),
                                bucket_counts=counts.tolist(),
                                global_step=step
                            )
                    except Exception as e:
                        print(f"  |- histogram write failed: {e}")
                    t_min = tensor_data.min().item()
                    t_max = tensor_data.max().item()
                    t_mean = tensor_data.mean().item()
                    shape_str = "x".join(map(str, tensor_data.shape))
                    print(f"  |- {dist_name}:")
                    print(f"  |   - Shape: [{shape_str}], Mean: {t_mean:.4f}, Min: {t_min:.4f}, Max: {t_max:.4f}")
                else:
                    print(f"  |- {dist_name}: contains NaN, skipped.")

        if step in args.save_iterations or (step == args.iterations and args.method in ("scaffoldgs", "octree_gs")):
            save_result = model.save(args.model_path, step)
            iteration_dir = os.path.join(args.model_path, "point_cloud", f"iteration_{step}")
            observed_assets = []
            observation_error = None
            if os.path.isdir(iteration_dir):
                try:
                    for directory, _, files in os.walk(iteration_dir, followlinks=False):
                        for name in sorted(files):
                            asset = os.path.join(directory, name)
                            observed_assets.append({"path": asset, "bytes": os.path.getsize(asset)})
                except OSError as exc:
                    observation_error = str(exc)
            checkpoint_event = {"step": step, "model_path": args.model_path,
                                "elapsed_wall_seconds_current_process": time.time() - t_train_start,
                                "observed_iteration_dir": iteration_dir if os.path.isdir(iteration_dir) else None,
                                "observed_assets": observed_assets,
                                "asset_set_complete": "unknown", "optimizer_state_present": "unknown",
                                "training_resume_verified": False, "observation_error": observation_error,
                                "wrapper_save_return": save_result}
            tb_writer.add_text("events/checkpoint_save_call_returned", json.dumps(checkpoint_event, default=str), step)
            tb_writer.flush()
            print(f"\n[CheckPoint] Assets saved to: {args.model_path} (Step {step})")
    t_train_end = time.time()
    total_training_seconds = t_train_end - t_train_start

    with open(timing_path, "w") as f:
        json.dump({
            "train_start_unix": t_train_start,
            "train_end_unix": t_train_end,
            "total_training_seconds": total_training_seconds,
            "total_training_minutes": total_training_seconds / 60.0,
            "cumulative_gpu_time_seconds": cumulative_gpu_time,
            "cumulative_gpu_time_scope": "sum of unsynchronized per-step wall times; not pure GPU time",
            "total_training_seconds_scope": "current process training loop including logging and checkpoint calls",
            "iterations": args.iterations,
            "status": "complete"
        }, f, indent=4)

    print(f"\n[Timing] Training complete. Total: {total_training_seconds:.1f}s / {total_training_seconds/60.0:.2f}min")
    print(f"[Timing] timing.json written to {timing_path}")

    tb_writer.close()
    if not args.skip_render:
        in_memory_render(model, args.model_path, args.iterations)

if __name__ == "__main__":
    main()
