import numpy as np

from .transformation import Transformation


class Identity(Transformation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x
