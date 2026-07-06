from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    name: str = "base"
    group: str = "other"
    supports_uncertainty: bool = False
    stochastic: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def set_seed(self, seed: int) -> "BaseModel":
        return self

    def predict_std(self, X: np.ndarray):
        return None
