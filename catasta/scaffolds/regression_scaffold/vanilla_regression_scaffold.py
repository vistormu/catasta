import numpy as np

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss

from vclog import Logger

from ...datasets import RegressionDataset
from ...entities import RegressionEvalInfo, RegressionPrediction, RegressionTrainInfo
from ...models import Regressor
from .use_cases import get_optimizer, get_loss_function


class VanillaRegressionScaffold:
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

        self.optimmizer_id: str = optimizer
        self.loss_function_id: str = loss_function

        self.logger: Logger = Logger("catasta")
        self.logger.info(f"using device: {self.device}")

        self.train_split: float = 0.8  # TMP

    def train(self, *,
              epochs: int = 100,
              batch_size: int = 128,
              train_split: float = 0.8,
              lr: float = 1e-3,
              ) -> RegressionTrainInfo:
        self.model.train()

        self.train_split = train_split  # TMP

        optimizer: Optimizer | None = get_optimizer(self.optimmizer_id, self.model, lr)
        if optimizer is None:
            raise ValueError(f"invalid optimizer id: {self.optimmizer_id}")

        loss_function: _Loss | None = get_loss_function(self.loss_function_id)
        if loss_function is None:
            raise ValueError(f"invalid loss function id: {self.loss_function_id}")

        train_dataset: Subset = Subset(self.dataset, range(int(len(self.dataset) * train_split)))
        val_dataset: Subset = Subset(self.dataset, range(int(len(self.dataset) * train_split), len(self.dataset)))
        data_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        best_loss: float = np.inf
        best_model_state_dict: dict = self.model.state_dict()

        train_losses: list[float] = []
        eval_losses: list[float] = []
        for i in range(epochs):
            batch_train_losses: list[float] = []
            for j, (x_batch, y_batch) in enumerate(data_loader):
                optimizer.zero_grad()

                x_batch: Tensor = x_batch.to(self.device, dtype=self.dtype)
                y_batch: Tensor = y_batch.to(self.device, dtype=self.dtype)

                output: Tensor = self.model(x_batch)

                loss: Tensor = loss_function(output, y_batch)
                loss.backward()

                optimizer.step()

                batch_train_losses.append(loss.item())

                self.logger.info(f"epoch {i}/{epochs} | {int((i/epochs)*100+(j/len(data_loader))*100/epochs)}% | train loss: {loss.item():.4f} | eval loss: {best_loss:.4f}", flush=True)

            train_losses.append(np.mean(batch_train_losses).astype(float))

            eval_loss: float = self._estimate_loss(val_dataset, batch_size=batch_size)
            eval_losses.append(eval_loss)

            if eval_loss < best_loss:
                best_loss = eval_loss
                best_model_state_dict = self.model.state_dict()

        self.logger.info(f'epoch {epochs}/{epochs} | 100% | loss: {np.min(eval_losses):.4f}')

        self.model.load_state_dict(best_model_state_dict)

        return RegressionTrainInfo(np.array(train_losses), np.array(eval_losses))

    @ torch.no_grad()
    def predict(self, input: np.ndarray | Tensor) -> RegressionPrediction:
        self.model.eval()

        input_tensor: Tensor = torch.tensor(input) if isinstance(input, np.ndarray) else input
        input_tensor = input_tensor.to(self.device, dtype=self.dtype)

        output: Tensor = self.model(input_tensor)

        return RegressionPrediction(output.cpu().numpy())

    @ torch.no_grad()
    def evaluate(self) -> RegressionEvalInfo:
        test_x: np.ndarray = self.dataset.inputs[int(len(self.dataset) * self.train_split)+1:]
        test_y: np.ndarray = self.dataset.outputs[int(len(self.dataset) * self.train_split)+1:]

        test_dataset = Subset(self.dataset, range(int(len(self.dataset) * self.train_split)+1, len(self.dataset)))
        data_loader: DataLoader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        predictions: np.ndarray = np.array([])
        for x_batch, _ in data_loader:
            output: RegressionPrediction = self.predict(x_batch)
            predictions = np.concatenate((predictions, output.prediction.flatten()))

        return RegressionEvalInfo(test_x, test_y, predictions)

    @ torch.no_grad()
    def _estimate_loss(self, val_dataset: Dataset, batch_size: int) -> float:
        self.model.eval()

        loss_function: _Loss | None = get_loss_function(self.loss_function_id)
        if loss_function is None:
            raise ValueError(f"invalid loss function id: {self.loss_function_id}")

        data_loader: DataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        losses: list[float] = []
        for x_batch, y_batch in data_loader:
            x_batch: Tensor = x_batch.to(self.device, dtype=self.dtype)
            y_batch: Tensor = y_batch.to(self.device, dtype=self.dtype)

            output: Tensor = self.model(x_batch)

            loss: Tensor = loss_function(output, y_batch)

            losses.append(loss.item())

        self.model.train()

        return np.mean(losses).astype(float)
