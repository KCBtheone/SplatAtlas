from typing import Dict, Type
from .base_method import BaseMethod

METHOD_REGISTRY: Dict[str, Type[BaseMethod]] = {}

def register_method(name: str):
    def decorator(cls: Type[BaseMethod]):
        METHOD_REGISTRY[name] = cls
        return cls
    return decorator
