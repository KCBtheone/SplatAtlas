import importlib
from core.registry import METHOD_REGISTRY

SUPPORTED_METHODS = (
    '2dgs',
    '3dgs_dr',
    '3dgsmcmc',
    'absgs',
    'analyticsplatting',
    'atomgs',
    'conegs',
    'erankgs',
    'gaussianpro',
    'ges',
    'ghap',
    'gof',
    'gslpm',
    'hogs',
    'lapisgs',
    'lightgaussian',
    'minisplatting',
    'mipsplatting',
    'octree_gs',
    'op43dgs',
    'opti3dgs',
    'pgsr',
    'pixelgs',
    'pup3dgs',
    'radegs',
    'reactgs',
    'scaffoldgs',
    'sogs',
    'steepgs',
    'vanilla_3dgs',
)


def load_method(method_name: str):
    if method_name not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method {method_name!r}. Choose from: {', '.join(SUPPORTED_METHODS)}")
    module_name = 'wrapper_vanilla' if method_name == 'vanilla_3dgs' else f'wrapper_{method_name}'
    try:
        importlib.import_module(f'.{module_name}', package=__name__)
    except ImportError as error:
        raise RuntimeError(f"Could not import {module_name}: {error}") from error
    if method_name not in METHOD_REGISTRY:
        raise KeyError(f"{module_name} did not register {method_name!r}")
    return METHOD_REGISTRY[method_name]
