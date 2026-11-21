import argparse
import os
import json
from pathlib import Path
import sys

import pandas as pd

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.paths import OUTPUT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_ROOT)
    parser.add_argument("--iteration", type=int, default=30000)
    args = parser.parse_args()
    outputs_dir = args.output_dir
    data = []

    for folder in os.listdir(outputs_dir):
        method_dir = os.path.join(outputs_dir, folder)
        if not os.path.isdir(method_dir) or folder == "pathology_archive":
            continue

        metrics_file = os.path.join(method_dir, f"metrics_test_iter{args.iteration}.json")
        if not os.path.exists(metrics_file):
            metrics_file = os.path.join(method_dir, "metrics.json")

        train_metrics_file = os.path.join(method_dir, f"metrics_train_iter{args.iteration}.json")
        physics_file = os.path.join(method_dir, f"offline_physics_{args.iteration}.json")

        method, _, scene = folder.rpartition('_')
        row = {"Method": method or folder, "Scene": scene if method else "unknown"}

        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                m = json.load(f)
                row["Method"] = m.get("method", row["Method"])
                row["Scene"] = m.get("scene", row["Scene"])
                row["PSNR_Test"] = m.get("photometric", {}).get("PSNR", 0)
                row["SSIM"] = m.get("photometric", {}).get("SSIM", 0)
                row["Prior"] = m.get("prior_type", "none")
                row["Flags"] = len(m.get("pathology_flags", []))

                if os.path.exists(train_metrics_file):
                    with open(train_metrics_file, 'r') as tf:
                        tm = json.load(tf)
                        row["PSNR_Train"] = tm.get("photometric", {}).get("PSNR", 0)
                        row["Delta_PSNR"] = row["PSNR_Train"] - row["PSNR_Test"]

        if os.path.exists(physics_file):
            with open(physics_file, 'r') as f:
                p = json.load(f)
                row["Gini"] = p.get("spatial_gini", 0)
                row["SH_Ratio"] = p.get("sh_energy_ratio", 0)
                row["Condition"] = p.get("scale_condition_mean", 0)
                row["Billboard_Bias"] = p.get("billboard_bias_ratio", 0)

        if len(row) > 2:
            data.append(row)

    if not data:
        return

    df = pd.DataFrame(data).round(3)
    csv_path = os.path.join(outputs_dir, "grand_leaderboard.csv")
    df.to_csv(csv_path, index=False)

    latex_str = df.to_latex(index=False, float_format="%.3f")
    with open(os.path.join(outputs_dir, "grand_leaderboard.tex"), "w") as f:
        f.write(latex_str)

    print(f"[GrandTable] 大榜单生成完毕！共收录 {len(data)} 个模型数据。")

if __name__ == "__main__":
    main()
