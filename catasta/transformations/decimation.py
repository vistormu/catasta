import numpy as np

from .transformation import Transformation


class Decimation(Transformation):
    def __init__(self, decimation_factor: int) -> None:
        """Initialize the Decimation object.

        Arguments
        ---------
        decimation_factor : int
            The factor by which to decimate the data.
        """
        self.decimation_factor: int = decimation_factor

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Decimate the data.

        Arguments
        ---------
        x : np.ndarray
            The data to decimate.

        Returns
        -------
        np.ndarray
            The decimated data.
        """
        return x[::self.decimation_factor]
