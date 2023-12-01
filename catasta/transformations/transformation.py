import numpy as np


class Transformation:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
