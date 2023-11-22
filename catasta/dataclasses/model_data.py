from typing import NamedTuple
import numpy as np


class ModelData(NamedTuple):
    input: np.ndarray
    output: np.ndarray
