import numpy as np

from .transformation import Transformation


class Normalization(Transformation):
    def __init__(self, norm_tech: str) -> None:
        self.norm_tech: str = norm_tech

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.norm_tech == "minmax":
            return (x - np.min(x)) / (np.max(x) - np.min(x))
        elif self.norm_tech == "zscore":
            return (x - np.mean(x)) / np.std(x)
        else:
            raise ValueError("Unknown normalization technique")
