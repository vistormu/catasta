from typing import NamedTuple
import numpy as np


class ClassificationPrediction(NamedTuple):
    value: np.ndarray
    argmax: np.ndarray
