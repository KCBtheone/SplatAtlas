import argparse
import os
from pathlib import Path
import subprocess
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import paths
from methods import SUPPORTED_METHODS


def run_command(script, arguments):
    command = [sys.executable, os.path.join(paths.PROJECT_ROOT, script), *arguments]
    print(f"\n[EXEC] {command}", flush=True)
    return subprocess.run(command, cwd=paths.PROJECT_ROOT).returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(paths.OUTPUT_ROOT))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = paths.load_yaml(stream)

    iters = str(config["global"]["iterations"])

    for task in config["tasks"]:
        method = task["method"]
        if method not in SUPPORTED_METHODS:
            parser.error(f"Unknown method {method!r}")
        for scene in task["scenes"]:
            source_path = Path(task["base_path"]).expanduser().resolve() / scene
            model_path = args.output_dir.expanduser().resolve() / f"{method}_{scene}"
            inputs = ["--method", method, "--source_path", str(source_path), "--model_path", str(model_path)]
            grid = ["--resolution", str(task["resolution"])] if "resolution" in task else []

            if not run_command("scripts/main_train.py", inputs + grid + [
                "--iterations", iters, "--save_iterations", iters, "--track_decoupling", "--skip_render",
            ]):
                print(f"Training failed: {method}/{scene}")
                continue

            if not run_command("scripts/main_render.py", inputs + grid + ["--iteration", iters]):
                print(f"Rendering failed: {method}/{scene}")
                continue

            if not run_command("ufd_evalkit/run_eval.py", [
                "--method", method, "--scene", scene, "--colmap_dir", str(source_path),
                "--render_dir", str(model_path / f"renders_test_{iters}"),
                "--gt_dir", str(model_path / f"gt_test_{iters}"),
                "--ply_path", str(model_path / "point_cloud" / f"iteration_{iters}" / "point_cloud.ply"),
                "--output_json", str(model_path / f"metrics_test_iter{iters}.json"),
            ]):
                print(f"Evaluation failed: {method}/{scene}")
                continue

            if not run_command("scripts/compute_offline_physics.py", inputs + ["--iteration", iters]):
                print(f"Physical diagnostics failed: {method}/{scene}")
                continue

            if run_command("scripts/build_grand_table.py", ["--output-dir", str(model_path.parent), "--iteration", iters]):
                print(f"Completed: {method}/{scene}")


if __name__ == "__main__":
    main()
