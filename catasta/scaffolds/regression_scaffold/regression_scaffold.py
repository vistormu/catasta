import numpy as np

import torch
from torch import Tensor

from ...datasets import RegressionDataset
from ...entities import RegressionEvalInfo, RegressionPrediction, RegressionTrainInfo
from ...models import Regressor, GaussianRegressor, VanillaRegressor
from .vanilla_regression_scaffold import VanillaRegressionScaffold
from .gaussian_regression_scaffold import GaussianRegressionScaffold


class RegressionScaffold:
    '''
    A scaffold for training a regression model.

    Methods
    -------
    train -> ~catasta.entities.RegressionTrainInfo
        Train the model.
    predict -> ~catasta.entities.RegressionPrediction
        Predict the output for the given input.
    evaluate -> ~catasta.entities.RegressionEvalInfo
        Evaluate the model on the test set.
    '''

    def __init__(self, *,
                 model: Regressor,
                 dataset: RegressionDataset,
                 optimizer: str = "adam",
                 loss_function: str = "mse",
                 ) -> None:
        '''
        Parameters
        ----------
        model : ~catasta.model.Regressor
            The model to train.
        dataset : ~catasta.dataset.RegressionDataset
            The dataset to train on.
        optimizer : str, optional
            The optimizer to use, by default "adam". Possible values are "adam", "sgd" and "adamw".
        loss_function : str, optional
            The loss function to use, by default "mse". Possible values are "mse", "l1", "smooth_l1" and "huber".

        Raises
        ------
        ValueError
            If an invalid optimizer or loss function is specified.
        '''
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype: torch.dtype = torch.float32

        self.model: Regressor = model.to(self.device)
        self.dataset: RegressionDataset = dataset

        if isinstance(model, VanillaRegressor):
            self.scaffold: VanillaRegressionScaffold = VanillaRegressionScaffold(
                model=model,
                dataset=dataset,
                optimizer=optimizer,
                loss_function=loss_function,
            )
        elif isinstance(model, GaussianRegressor):
            self.scaffold: GaussianRegressionScaffold = GaussianRegressionScaffold(
                model=model,
                dataset=dataset,
            )

    def train(self, *,
              epochs: int = 100,
              batch_size: int = 128,
              train_split: float = 0.8,
              lr: float = 1e-3,
              ) -> RegressionTrainInfo:
        return self.scaffold.train(epochs=epochs, batch_size=batch_size, train_split=train_split, lr=lr)

    @ torch.no_grad()
    def predict(self, input: np.ndarray | Tensor) -> RegressionPrediction:
        return self.scaffold.predict(input)

    @ torch.no_grad()
    def evaluate(self) -> RegressionEvalInfo:
        return self.scaffold.evaluate()
