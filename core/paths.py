import os
from pathlib import Path
import re
import shutil
import sys


def _root(variable, default):
    value = os.environ.get(variable) or default
    return Path(value).expanduser().resolve().as_posix()


PROJECT_ROOT = _root("SPLATATLAS_ROOT", Path(__file__).resolve().parents[1])
WORK_ROOT = _root("SPLATATLAS_WORK_ROOT", PROJECT_ROOT)
DATA_ROOT = _root("SPLATATLAS_DATA_ROOT", Path(WORK_ROOT) / "datasets")
OUTPUT_ROOT = _root("SPLATATLAS_OUTPUT_ROOT", Path(WORK_ROOT) / "outputs")
UPSTREAM_ROOT = _root("SPLATATLAS_UPSTREAM_ROOT", Path(WORK_ROOT) / "upstream")
ENV_ROOT = _root("SPLATATLAS_ENV_ROOT", Path(WORK_ROOT) / "envs")
TEMP_ROOT = _root("SPLATATLAS_TEMP_DIR", Path(WORK_ROOT) / "tmp")
CACHE_ROOT = _root("SPLATATLAS_CACHE_ROOT", Path(WORK_ROOT) / ".cache")

_conda_executable = os.environ.get("CONDA_EXE") or shutil.which("conda")
_conda_default = Path(_conda_executable).resolve().parents[1] if _conda_executable else sys.prefix
CONDA_ROOT = _root("SPLATATLAS_CONDA_ROOT", _conda_default)

_nvcc = shutil.which("nvcc")
_cuda_default = Path(_nvcc).resolve().parents[1] if _nvcc else Path(ENV_ROOT) / "cuda"
CUDA_ROOT = _root("CUDA_HOME", os.environ.get("CUDA_PATH") or _cuda_default)

_VARIABLES = {
    "SPLATATLAS_ROOT": PROJECT_ROOT,
    "SPLATATLAS_WORK_ROOT": WORK_ROOT,
    "SPLATATLAS_DATA_ROOT": DATA_ROOT,
    "SPLATATLAS_OUTPUT_ROOT": OUTPUT_ROOT,
    "SPLATATLAS_UPSTREAM_ROOT": UPSTREAM_ROOT,
    "SPLATATLAS_ENV_ROOT": ENV_ROOT,
    "SPLATATLAS_TEMP_DIR": TEMP_ROOT,
    "SPLATATLAS_CACHE_ROOT": CACHE_ROOT,
    "SPLATATLAS_CONDA_ROOT": CONDA_ROOT,
    "CUDA_HOME": CUDA_ROOT,
}
_VARIABLE_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}")


def expand_path_variables(value):
    return _VARIABLE_PATTERN.sub(lambda match: _VARIABLES.get(
        match.group(1), os.environ.get(match.group(1), match.group(0))
    ), value)


def load_yaml(source):
    import yaml

    def expand(value):
        if isinstance(value, str):
            return expand_path_variables(value)
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        return value

    return expand(yaml.safe_load(source))
