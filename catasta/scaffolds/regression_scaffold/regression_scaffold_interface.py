from abc import ABC, abstractmethod
import numpy as np
from torch import Tensor
from ...dataclasses import RegressionEvalInfo, RegressionPrediction, RegressionTrainInfo


class RegressionScaffold(ABC):
    @abstractmethod
    def train(self, *,
              epochs: int,
              batch_size: int,
              lr: float,
              final_lr: float | None = None,
              early_stopping: tuple[int, float] | None = None,
              verbose: bool = True,
              ) -> RegressionTrainInfo:
        """
        Train the model

        Arguments
        ---------
        epochs: int
            Number of epochs to train for
        batch_size: int
            Batch size
        lr: float
            Learning rate
        final_lr: float
            Final learning rate for the learning rate scheduler. If None, then the learning rate will not decay.
        early_stopping: tuple[int, float]
            A tuple of (patience, min_delta) for the early stopping scheduler. If None, then the early stopping scheduler will not be used.
        verbose: bool
            Whether to print training logs

        Returns
        -------
        RegressionTrainInfo
            Training information
        """

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

    @abstractmethod
    def evaluate(self) -> RegressionEvalInfo:
        """
        Evaluate the model

        Returns
        -------
        RegressionEvalInfo
            Evaluation information
        """
