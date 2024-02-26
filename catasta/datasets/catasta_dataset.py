from abc import ABC

import numpy as np

from torch.utils.data import Subset

from ..transformations import Transformation


class CatastaDataset(ABC):
    def __init__(self, *,
                 root: str,
                 input_transformations: list[Transformation],
                 output_transformations: list[Transformation],
                 splits: tuple[float, float, float],
                 ) -> None:
        self.root: str = root
        self.train_split: float = splits[0]
        self.validation_split: float = splits[1]
        self.test_split: float = splits[2]

        self.input_transformations: list[Transformation] = input_transformations
        self.output_transformations: list[Transformation] = output_transformations

        self.inputs: np.ndarray = np.array([])
        self.outputs: np.ndarray = np.array([])

        self.train: Subset = None  # type: ignore
        self.validation: Subset | None = None
        self.test: Subset | None = None
