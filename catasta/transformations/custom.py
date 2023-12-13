from typing import Callable

import numpy as np

from .transformation import Transformation


class Custom(Transformation):
    def __init__(self, function: Callable) -> None:
        """
        Initialize the custom transformation

        Parameters
        ----------
        function : Callable
            Custom transformation function
        """
        self.function: Callable = function

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply custom transformation to the input data

        Parameters
        ----------
        x : np.ndarray
            Input data to be transformed

        Returns
        -------
        np.ndarray
            Transformed data
        """
        return self.function(x)
