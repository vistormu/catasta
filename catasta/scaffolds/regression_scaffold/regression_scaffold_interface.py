from abc import ABC, abstractmethod
import numpy as np
from torch import Tensor
from ...entities import RegressionEvalInfo, RegressionPrediction, RegressionTrainInfo


class IRegressionScaffold(ABC):
    @abstractmethod
    def train(self, *,
              epochs: int,
              batch_size: int,
              lr: float,
              ) -> RegressionTrainInfo:
        pass

    @abstractmethod
    def predict(self, input: np.ndarray | Tensor) -> RegressionPrediction:
        pass

    @abstractmethod
    def evaluate(self) -> RegressionEvalInfo:
        pass
