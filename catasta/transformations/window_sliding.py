import numpy as np

from .transformation import Transformation


class WindowSliding(Transformation):
    def __init__(self, window_size: int, stride: int) -> None:
        self.window_size: int = window_size
        self.stride: int = stride

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                x[i: i + self.window_size]
                for i in range(0, len(x) - self.window_size + 1, self.stride)
            ]
        )
