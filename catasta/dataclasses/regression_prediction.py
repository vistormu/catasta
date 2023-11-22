from typing import NamedTuple
import numpy as np


class RegressionPrediction(NamedTuple):
    prediction: np.ndarray
    stds: np.ndarray | None = None
