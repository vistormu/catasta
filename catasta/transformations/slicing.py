import numpy as np

from .transformation import Transformation


class Slicing(Transformation):
    def __init__(self, amount: int, end: str) -> None:
        self.amount: int = amount
        self.end: str = end

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.end == "left":
            return x[self.amount:]
        elif self.end == "right":
            return x[:-self.amount]
        else:
            raise ValueError("Unknown slicing end")
