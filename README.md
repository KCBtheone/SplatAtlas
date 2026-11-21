# SplatAtlas

SplatAtlas is a benchmark and research framework for comparing the scene representations learned by Gaussian splatting methods. The paper studies 30 methods on 31 shared scenes across four datasets. The benchmark connects novel-view synthesis scores to saved model states and their rendering conditions so researchers can compare how different methods represent the same scene.

The framework provides common entry points for training, saving and loading model states, rendering, and measurement. Method adapters call their upstream implementations in compatible environments. The research resources include training trajectories, intermediate checkpoints, and native model assets for studying primitive counts, opacity, shape, and spatial allocation. Where the representation and renderer support it, saved states can also be used to measure rendering contributions and responses to edits such as primitive deletion.

The experiments reported in the paper are complete. Additional runs are underway, and we are checking their setups and results for correctness.

- [Benchmark results and paper](https://kcbtheone.github.io/SplatAtlas/)
- [Result tables and archive manifest](https://huggingface.co/datasets/KCBtheone/splatatlas-core)
- [Model and experiment archives](https://huggingface.co/datasets/KCBtheone/splatatlas-runs)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Use the framework

This release includes adapters for the 30 methods in the paper. Each adapter uses the corresponding upstream setup and its Python and CUDA dependencies.

| Task | Entry point |
| --- | --- |
| Train a method and save selected iterations | [`scripts/main_train.py`](scripts/main_train.py) |
| Load a saved model and render scene views | [`scripts/main_render.py`](scripts/main_render.py) |
| Measure image quality and supported model diagnostics | [`ufd_evalkit/`](ufd_evalkit/) |
| Run a configured batch in the selected method environment | [`scripts/run_pipeline.py`](scripts/run_pipeline.py), [`batch_example.yaml`](configs/batch_example.yaml) |
| Integrate a method with its native assets and renderer | [`core/base_method.py`](core/base_method.py), [`methods/`](methods/) |

To work from an existing result, select a method and scene in the benchmark and use the archive manifest to locate the model assets. Load the recorded iteration with the corresponding scene data, camera split, resolution, and method configuration. Preserve any method-specific weights or native files required by the adapter. For paper comparisons, use the evaluation settings associated with the selected result.

To try the batch runner, edit `configs/batch_example.yaml` to point to your scene data and run `python scripts/run_pipeline.py --config configs/batch_example.yaml` in the selected method’s environment. The other YAML files in `configs/` show example scene settings for individual methods. Use the benchmark records for the settings associated with a specific paper result.

## Paths

Run the Python entry points in the environment required by the chosen method. Set these variables before starting a process to use a different directory.

| Variable | Default |
| --- | --- |
| `SPLATATLAS_ROOT` | Directory containing this README |
| `SPLATATLAS_WORK_ROOT` | Project root |
| `SPLATATLAS_DATA_ROOT` | `datasets/` under the work root |
| `SPLATATLAS_UPSTREAM_ROOT` | `upstream/` under the work root |
| `SPLATATLAS_OUTPUT_ROOT` | `outputs/` under the work root |
| `SPLATATLAS_ENV_ROOT` | `envs/` under the work root |
| `SPLATATLAS_TEMP_DIR` | `tmp/` under the work root |
| `SPLATATLAS_CACHE_ROOT` | `.cache/` under the work root |

Place each upstream repository under the upstream root using the directory name referenced by its adapter. Follow the upstream project's setup instructions for Python packages and CUDA extensions. Build scripts use the active Python or `SPLATATLAS_BUILD_PYTHON`. Set `CUDA_HOME`, `CC`, `CXX`, and `TORCH_CUDA_ARCH_LIST` if the build requires explicit choices.
