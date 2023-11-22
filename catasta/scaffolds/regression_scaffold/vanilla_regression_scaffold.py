import time

import numpy as np

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.nn.modules.loss import _Loss

from vclog import Logger

from .regression_scaffold_interface import IRegressionScaffold
from ...datasets import RegressionDataset
from ...entities import RegressionEvalInfo, RegressionPrediction, RegressionTrainInfo
from .use_cases import get_optimizer, get_loss_function, log_train_data


class VanillaRegressionScaffold(IRegressionScaffold):
    def __init__(self, *,
                 model: Module,
                 dataset: RegressionDataset,
                 optimizer: str = "adam",
                 loss_function: str = "mse",
                 ) -> None:
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype: torch.dtype = torch.float32

        self.model: Module = model.to(self.device)
        self.dataset: RegressionDataset = dataset

        self.optimmizer_id: str = optimizer
        self.loss_function_id: str = loss_function

        self.logger: Logger = Logger("catasta")

        # Logging info
        message: str = f"using {self.device} with {torch.cuda.get_device_name()}" if torch.cuda.is_available() else f"using {self.device}"
        self.logger.info(message)
        n_parameters: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"training model {self.model.__class__.__name__} with {n_parameters} parameters")

    def train(self, *,
              epochs: int = 100,
              batch_size: int = 128,
              lr: float = 1e-3,
              final_lr: float | None = None,
              ) -> RegressionTrainInfo:
        self.model.train()

        optimizer: Optimizer | None = get_optimizer(self.optimmizer_id, self.model, lr)
        if optimizer is None:
            raise ValueError(f"invalid optimizer id: {self.optimmizer_id}")

        loss_function: _Loss | None = get_loss_function(self.loss_function_id)
        if loss_function is None:
            raise ValueError(f"invalid loss function id: {self.loss_function_id}")

        lr_decay: float = (final_lr / lr) ** (1 / epochs) if final_lr is not None else 1.0
        scheduler: StepLR = StepLR(optimizer, step_size=1, gamma=lr_decay)

        data_loader: DataLoader = DataLoader(self.dataset.train, batch_size=batch_size, shuffle=False)

        best_eval_loss: float = np.inf
        eval_loss: float = np.inf
        best_model_state_dict: dict = self.model.state_dict()

        time_per_batch: float = 0.0
        time_per_epoch: float = 0.0

        train_losses: list[float] = []
        eval_losses: list[float] = []
        for i in range(epochs):
            times_per_batch: list[float] = []
            batch_train_losses: list[float] = []
            for j, (x_batch, y_batch) in enumerate(data_loader):
                start_time: float = time.time()

                optimizer.zero_grad()

                x_batch: Tensor = x_batch.to(self.device, dtype=self.dtype)
                y_batch: Tensor = y_batch.to(self.device, dtype=self.dtype)

                output: Tensor = self.model(x_batch)

                loss: Tensor = loss_function(output, y_batch)
                loss.backward()

                optimizer.step()

                batch_train_losses.append(loss.item())
                times_per_batch.append((time.time() - start_time) * 1000)

                log_train_data(
                    train_loss=loss.item(),
                    val_loss=eval_loss,
                    best_val_loss=best_eval_loss,
                    lr=scheduler.get_last_lr()[0],
                    epoch=i,
                    epochs=epochs,
                    percentage=int((i/epochs)*100+(j/len(data_loader))*100/epochs),
                    time_per_batch=time_per_batch,
                    time_per_epoch=time_per_epoch,
                )

            # END OF EPOCH
            scheduler.step()

            train_losses.append(np.mean(batch_train_losses).astype(float))
            time_per_batch = np.mean(times_per_batch).astype(float)
            time_per_epoch = np.sum(times_per_batch).astype(float)

            eval_loss = self._estimate_loss(batch_size)
            eval_losses.append(eval_loss)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_model_state_dict = self.model.state_dict()

        # END OF TRAINING
        self.logger.info(f'epoch {epochs}/{epochs} | 100% | best eval loss: {np.min(eval_losses):.4f}')

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
        test_index: int = np.floor(len(self.dataset) * (self.dataset.splits[0]+self.dataset.splits[1])).astype(int)

        if self.dataset.test is None and self.dataset.validation is not None:
            Logger.warning("test split is empty, using validation split instead")
            self.dataset.test = self.dataset.validation
            test_index = np.floor(len(self.dataset) * self.dataset.splits[0]).astype(int)
        else:
            raise ValueError(f"cannot evaluate without a test split")

        test_x: np.ndarray = self.dataset.inputs[test_index:]
        test_y: np.ndarray = self.dataset.outputs[test_index:].flatten()

        data_loader: DataLoader = DataLoader(self.dataset.test, batch_size=1, shuffle=False)

        predictions: np.ndarray = np.array([])
        for x_batch, _ in data_loader:
            output: RegressionPrediction = self.predict(x_batch)
            predictions = np.concatenate((predictions, output.prediction.flatten()))

        if len(predictions) != len(test_y):
            min_len: int = min(len(predictions), len(test_y))
            predictions = predictions[-min_len:]
            test_y = test_y[-min_len:]

        return RegressionEvalInfo(test_x, test_y, predictions)

    @ torch.no_grad()
    def _estimate_loss(self, batch_size: int) -> float:
        if self.dataset.validation is None:
            return np.inf

        self.model.eval()

        loss_function: _Loss | None = get_loss_function(self.loss_function_id)
        if loss_function is None:
            raise ValueError(f"invalid loss function id: {self.loss_function_id}")

        data_loader: DataLoader = DataLoader(self.dataset.validation, batch_size=batch_size, shuffle=False)

        losses: list[float] = []
        for x_batch, y_batch in data_loader:
            x_batch: Tensor = x_batch.to(self.device, dtype=self.dtype)
            y_batch: Tensor = y_batch.to(self.device, dtype=self.dtype)

            output: Tensor = self.model(x_batch)

            loss: Tensor = loss_function(output, y_batch)

            losses.append(loss.item())

        self.model.train()

        return np.mean(losses).astype(float)
