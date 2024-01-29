from abc import ABC, abstractmethod
import numpy as np
from torch import Tensor
from ...dataclasses import RegressionPrediction


class RegressionArchway(ABC):
    @abstractmethod
    def predict(self, input: np.ndarray | Tensor) -> RegressionPrediction:
        """
        Make predictions

        Arguments
        ---------
        input: np.ndarray | Tensor
            Input data

        Returns
        -------
        RegressionPrediction
            Prediction information
        """
