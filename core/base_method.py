import abc
from typing import Dict, Any, List
import torch

class BaseMethod(abc.ABC):
    
    @abc.abstractmethod
    def __init__(self, dataset_config: Dict, hyperparams: Dict):
        pass

    @abc.abstractmethod
    def train_iteration(self, step: int) -> Dict[str, float]:
        pass

    @abc.abstractmethod
    def render(self, camera: Any) -> Dict[str, torch.Tensor]:
        pass

    @abc.abstractmethod
    def save(self, save_dir: str, step: int):
        pass

    @abc.abstractmethod
    def load(self, model_path: str, iteration: int):
        pass

    @abc.abstractmethod
    def get_spatial_centers(self) -> torch.Tensor:
        raise NotImplementedError("该 Wrapper 尚未实现 get_spatial_centers")

    @abc.abstractmethod
    def compute_physical_metrics(self, cameras: List[Any] = None) -> Dict[str, float]:
        raise NotImplementedError("该 Wrapper 尚未实现 compute_physical_metrics")
        
    @abc.abstractmethod
    def evaluate_spatial_field(self, query_points: torch.Tensor, cameras: List[Any] = None) -> torch.Tensor:
        raise NotImplementedError("该 Wrapper 尚未实现 evaluate_spatial_field")
