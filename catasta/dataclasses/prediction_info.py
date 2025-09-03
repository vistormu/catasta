from typing import NamedTuple
import numpy as np


class PredictionInfo(NamedTuple):
    """NamedTuple for storing prediction information.

    Attributes
    ----------
    value : np.ndarray
        The predicted values.
    argmax : np.ndarray
        The argmax of the predicted values. If the task is regression, this array will be empty.
    std : np.ndarray
        The standard deviation of the predicted values. If the task is regression, this array will be a vector of zeroes.
    """
    value: np.ndarray
    std: np.ndarray
    argmax: np.ndarray
