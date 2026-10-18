# SplatAtlas

[![Website](https://img.shields.io/badge/Website-Benchmark-315b8a?style=flat&logo=githubpages&logoColor=white)](https://kcbtheone.github.io/SplatAtlas/)
[![Hugging Face Data](https://img.shields.io/badge/Hugging_Face-Core_data-ffd21e?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/KCBtheone/splatatlas-core)
[![Hugging Face Run Archives](https://img.shields.io/badge/Hugging_Face-Run_archives-ffd21e?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/KCBtheone/splatatlas-runs)
[![Paper: Draft, not the official version](https://img.shields.io/badge/Paper-Draft-b31b1b?style=flat)](https://kcbtheone.github.io/SplatAtlas/SplatAtlas.pdf)

SplatAtlas provides a compact, table-based view of scene-level image-quality and geometry results for 3D Gaussian Splatting. Browse results by dataset or method, and select a method or value to inspect the underlying scenes.

## Data availability

**SplatAtlas data has been uploaded to Hugging Face.**

- [splatatlas-core](https://huggingface.co/datasets/KCBtheone/splatatlas-core): lightweight result tables, metadata, wrapper templates, and manifests.
- [splatatlas-runs](https://huggingface.co/datasets/KCBtheone/splatatlas-runs): large per-experiment archives, including exported Gaussian fields, checkpoints, and run records.

**SplatAtlasVol geometry evaluation is complete as of October 18, 2026.** All 308 selected method-scene records are project-evaluated: Tanks & Temples has 14 methods × 7 scenes (98/98), and DTU has 14 methods × 15 scans (210/210). F-score, precision, recall, accuracy, completeness, and CD are included; no published-paper values are used as placeholders.

The completed tables and per-cell provenance are available on the website and in this repository. Training, mesh readout, and alignment protocols remain method-specific; completion does not establish a uniform-training or strict original-paper reproduction. Full SplatAtlasVol run archives will be linked here when publicly released.

The [manuscript PDF](static/SplatAtlas.pdf) is a draft and is not the official version.

## Website

The website is published at [kcbtheone.github.io/SplatAtlas](https://kcbtheone.github.io/SplatAtlas/). GitHub Actions builds and deploys the site when changes are pushed to `main`.

The repository contains the website and its displayed result snapshots. Large research artifacts are hosted on Hugging Face.

## Local development

Use Node.js 22.13 or newer. The static build uses only the Node.js standard library.

```sh
npm run build
npm run dev
```

Open `http://127.0.0.1:4173`. Set `PORT` to use another local port.

## Result tables

- `data/benchmark.json`: SplatAtlas per-scene PSNR, SSIM, and LPIPS, with dataset means over available complete scene records.
- `data/vol-benchmark.json`: completed SplatAtlasVol Tanks & Temples F-score, precision (P), and recall (R); fractional values are displayed as percentages.
- `data/vol-dtu-benchmark.json`: completed SplatAtlasVol DTU accuracy (A), completeness (C), and CD in millimeters, with the accepted camera-matched SVRaster rerun.

Unavailable or unverified values remain blank; missing cells are not imputed.

The SplatAtlasVol JSON files are accepted per-cell release snapshots. Each score record retains its project-evaluation status, logical source reference, protocol, and applicable input revision. Keep this provenance when refreshing results; the legacy CSV helper does not regenerate the completed October 18 selection.
