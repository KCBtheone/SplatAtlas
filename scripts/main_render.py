if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import numpy as np
from tqdm import tqdm
import sys
import concurrent.futures

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import methods

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=methods.SUPPORTED_METHODS, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--images", type=str, default="images")
    return parser.parse_args()

def save_np(data, path): np.save(path, data)

def main():
    args = parse_args()
    import torch
    import torchvision
    from ufd_evalkit.geometric import depth_to_normal

    model_class = methods.load_method(args.method)

    dataset_config = {"source_path": args.source_path, "model_path": args.model_path, "resolution": args.resolution, "images": args.images}
    model = model_class(dataset_config, hyperparams={})

    print(f"Loading {args.method} checkpoint at iteration {args.iteration}")
    try:
        model.load(args.model_path, args.iteration)
    except Exception as error:
        raise RuntimeError(f"Could not load {args.method} checkpoint from {args.model_path}") from error

    print("Rendering checkpoint")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
    futures = []

    test_render_dir = os.path.join(args.model_path, f"renders_test_{args.iteration}")
    test_depth_dir = os.path.join(args.model_path, f"depths_test_{args.iteration}")
    test_normal_dir = os.path.join(args.model_path, f"normals_test_{args.iteration}")
    test_gt_dir = os.path.join(args.model_path, f"gt_test_{args.iteration}")
    for d in [test_render_dir, test_depth_dir, test_normal_dir, test_gt_dir]: os.makedirs(d, exist_ok=True)

    train_render_dir = os.path.join(args.model_path, f"renders_train_{args.iteration}")
    for d in [train_render_dir]: os.makedirs(d, exist_ok=True)

    with torch.no_grad():
        test_cams = model.scene.getTestCameras()
        if test_cams and isinstance(test_cams[0], list):
            flat_cams = [c for sub in test_cams for c in sub]
        else:
            flat_cams = test_cams
        for idx, view in enumerate(tqdm(flat_cams, desc="Rendering Test")):
            render_pkg = model.render(view)
            rendering = render_pkg.get("render", render_pkg.get("image")).detach().cpu()
            depth = render_pkg.get("depth", None)

            futures.append(executor.submit(torchvision.utils.save_image, rendering, os.path.join(test_render_dir, f'{idx:05d}.png')))
            futures.append(executor.submit(torchvision.utils.save_image, view.original_image.cpu(), os.path.join(test_gt_dir, f'{idx:05d}.png')))

            if depth is not None:
                depth_cpu = depth.detach().cpu()
                normal = render_pkg.get("normal", depth_to_normal(depth)).detach().cpu()
                futures.append(executor.submit(save_np, depth_cpu.numpy(), os.path.join(test_depth_dir, f'{idx:05d}.npy')))
                futures.append(executor.submit(torchvision.utils.save_image, normal, os.path.join(test_normal_dir, f'{idx:05d}.png')))

        for idx, view in enumerate(tqdm(model.scene.getTrainCameras(), desc="Rendering Train")):
            render_pkg = model.render(view)
            rendering = render_pkg.get("render", render_pkg.get("image")).detach().cpu()
            futures.append(executor.submit(torchvision.utils.save_image, rendering, os.path.join(train_render_dir, f'{idx:05d}.png')))

    executor.shutdown()
    for future in futures:
        future.result()

    flag_path = os.path.join(args.model_path, f"render_complete_{args.iteration}.flag")
    with open(flag_path, "w") as f: f.write("Completed by Async Engine")
    print(f"Render complete at iteration {args.iteration}")

if __name__ == "__main__":
    main()
