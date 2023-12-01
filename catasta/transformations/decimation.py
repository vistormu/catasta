import numpy as np

from .transformation import Transformation


class Decimation(Transformation):
    def __init__(self, decimation_factor: int) -> None:
        self.decimation_factor: int = decimation_factor

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x[::self.decimation_factor]
