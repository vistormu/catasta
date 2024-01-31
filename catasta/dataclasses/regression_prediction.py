from typing import NamedTuple
import numpy as np


class RegressionPrediction(NamedTuple):
    value: np.ndarray
    std: np.ndarray | None = None
