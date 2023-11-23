from abc import ABC, abstractmethod
import numpy as np
from torch import Tensor
from ...dataclasses import RegressionEvalInfo, RegressionPrediction, RegressionTrainInfo


class IRegressionScaffold(ABC):
    @abstractmethod
    def train(self, *,
              epochs: int,
              batch_size: int,
              lr: float,
              final_lr: float | None = None,
              early_stopping: bool = False,
              ) -> RegressionTrainInfo:
        pass

    @abstractmethod
    def predict(self, input: np.ndarray | Tensor) -> RegressionPrediction:
        pass

    @abstractmethod
    def evaluate(self) -> RegressionEvalInfo:
        pass
