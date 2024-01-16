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

from .regression_scaffold_interface import RegressionScaffold
from ...datasets import RegressionDataset
from ...dataclasses import RegressionEvalInfo, RegressionPrediction, RegressionTrainInfo
from ...utils import get_optimizer, get_loss_function, log_train_data, ModelStateManager


class VanillaRegressionScaffold(RegressionScaffold):
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
              early_stopping: tuple[int, float] | None = None,
              verbose: bool = True,
              ) -> RegressionTrainInfo:
        self.model.train()

        if self.dataset.validation is None:
            self.logger.warning("no validation split found")

        optimizer: Optimizer | None = get_optimizer(self.optimmizer_id, self.model, lr)
        if optimizer is None:
            raise ValueError(f"invalid optimizer id: {self.optimmizer_id}")

        loss_function: _Loss | None = get_loss_function(self.loss_function_id)
        if loss_function is None:
            raise ValueError(f"invalid loss function id: {self.loss_function_id}")

        lr_decay: float = (final_lr / lr) ** (1 / epochs) if final_lr is not None else 1.0
        scheduler: StepLR = StepLR(optimizer, step_size=1, gamma=lr_decay)

        model_state_manager = ModelStateManager(early_stopping)

        data_loader: DataLoader = DataLoader(self.dataset.train, batch_size=batch_size, shuffle=True)

        eval_loss: float = np.inf

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

                x_batch = x_batch.to(self.device, dtype=self.dtype)
                y_batch = y_batch.to(self.device, dtype=self.dtype)

                output: Tensor = self.model(x_batch)

                loss: Tensor = loss_function(output, y_batch)
                loss.backward()

                optimizer.step()

                batch_train_losses.append(loss.item())
                times_per_batch.append((time.time() - start_time) * 1000)

                if verbose:
                    log_train_data(
                        train_loss=train_losses[-1] if len(train_losses) > 0 else 0,
                        val_loss=eval_loss,
                        best_val_loss=model_state_manager.best_loss,
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

            model_state_manager(self.model.state_dict(), eval_loss)

            if model_state_manager.stop():
                self.logger.warning("early stopping")
                break

        # END OF TRAINING
        self.logger.info(f'training completed | best eval loss: {np.min(eval_losses):.4f}')

        self.model.load_state_dict(model_state_manager.get_best_model_state())

        return RegressionTrainInfo(np.array(train_losses), np.array(eval_losses))

    @torch.no_grad()
    def predict(self, input: np.ndarray | Tensor) -> RegressionPrediction:
        self.model.eval()

        input_tensor: Tensor = torch.tensor(input) if isinstance(input, np.ndarray) else input
        input_tensor = input_tensor.to(self.device, dtype=self.dtype)

        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)

        output: Tensor = self.model(input_tensor)

        return RegressionPrediction(output.cpu().numpy())

    @torch.no_grad()
    def evaluate(self) -> RegressionEvalInfo:
        if self.dataset.test is None and self.dataset.validation is not None:
            self.logger.warning("no test split found, using validation split for evaluation")
            self.dataset.test = self.dataset.validation
        else:
            raise ValueError(f"cannot evaluate without a test split")

        true_input: np.ndarray = np.array([])
        true_output: np.ndarray = np.array([])
        predicted_output: np.ndarray = np.array([])
        for x, y in self.dataset.test:
            output: RegressionPrediction = self.predict(x)

            true_input = np.append(true_input, x[-1])
            true_output = np.append(true_output, y)
            predicted_output = np.append(predicted_output, output.prediction)

        return RegressionEvalInfo(true_input, true_output, predicted_output)

    @torch.no_grad()
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
