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

**SplatAtlasVol is still in progress. Its data will be uploaded to Hugging Face when the benchmark is complete.** The website currently shows preliminary Tanks & Temples and DTU snapshots. Some values are being checked or rerun, so these tables should not be treated as final results.

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
- `data/vol-benchmark.json`: preliminary SplatAtlasVol Tanks & Temples F-score, precision (P), and recall (R).
- `data/vol-dtu-benchmark.json`: preliminary SplatAtlasVol DTU accuracy (A), completeness (C), and CD in millimeters.

Unavailable or unverified values remain blank; missing cells are not imputed.

The optional data-preparation script regenerates the SplatAtlas NVS and SplatAtlasVol TNT JSON files from source CSVs:

```sh
node scripts/build-site-data.mjs --nvs <nvs_cells_core.csv> --scenes <scene_inventory.csv> --vol <vol_tnt_geometry_selected.csv>
```

The DTU JSON is maintained as a separate selected-result snapshot.
