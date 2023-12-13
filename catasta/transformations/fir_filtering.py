import numpy as np
from scipy.signal import firwin, lfilter

from .transformation import Transformation


class FIRFiltering(Transformation):
    def __init__(self, fs: int, cutoff: float, numtaps: int) -> None:
        self.fs: int = fs
        self.cutoff: float = cutoff
        self.numtaps: int = numtaps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        fir_coeff: np.ndarray = firwin(
            numtaps=self.numtaps,
            cutoff=self.cutoff,
            fs=self.fs,
            window="hamming",
        )
        return np.array(lfilter(fir_coeff, 1.0, x))
