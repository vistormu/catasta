from abc import ABC, abstractmethod
from ...dataclasses import RegressionEvalInfo, RegressionTrainInfo


class RegressionScaffold(ABC):
    @abstractmethod
    def train(self, *,
              epochs: int,
              batch_size: int,
              lr: float,
              final_lr: float | None = None,
              early_stopping: tuple[int, float] | None = None,
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

        Returns
        -------
        RegressionTrainInfo
            Training information
        """

    @abstractmethod
    def evaluate(self) -> RegressionEvalInfo:
        """
        Evaluate the model with the test split

        Returns
        -------
        RegressionEvalInfo
            Evaluation information
        """
