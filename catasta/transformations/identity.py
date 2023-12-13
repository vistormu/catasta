import numpy as np


class Identity:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x
