import platform
import time
import os

import numpy as np

import torch
from torch import Tensor
from torch.nn import Module, Identity
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Distribution

from gpytorch.models.gp import GP
from gpytorch.likelihoods import GaussianLikelihood

from ..datasets import CatastaDataset
from .utils import (
    get_optimizer,
    get_loss_function,
    ModelStateManager,
    TrainingLogger,
)
from ..dataclasses import (
    TrainInfo,
    RegressionEvalInfo,
    ClassificationEvalInfo,
)

from vclog import Logger


def _get_device(device: str) -> torch.device:
    match device:
        case "cpu":
            return torch.device("cpu")
        case "cuda":
            return torch.device("cuda")
        case "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        case _:
            raise ValueError("Invalid device")


def _get_dtype(dtype: str) -> torch.dtype:
    match dtype:
        case "float16":
            return torch.float16
        case "float32":
            return torch.float32
        case "float64":
            return torch.float64
        case _:
            raise ValueError("Invalid dtype")


class Scaffold:
    def __init__(self, *,
                 model: Module,
                 dataset: CatastaDataset,
                 optimizer: str,
                 loss_function: str | Module,
                 probabilistic: bool = False,
                 device: str = "auto",
                 dtype: str = "float32",
                 ) -> None:
        """The `Scaffold` class is a high-level API for training models on Catasta datasets.

        Arguments
        ---------
        model : torch.nn.Module
            The model to be trained.
        dataset : ~catasta.datasets.CatastaDataset
            The dataset to be used for training.
        """
        self._init(model, dataset, optimizer, loss_function, probabilistic, device, dtype)

    def train(self, *,
              epochs: int,
              batch_size: int,
              lr: float,
              final_lr: float | None = None,
              early_stopping: tuple[int, float] | None = None,
              shuffle: bool = True,
              ) -> TrainInfo:
        """Train the model on the dataset.

        Arguments
        ---------
        epochs : int
            The number of epochs to train the model.
        """
        return self._train(epochs, batch_size, lr, final_lr, early_stopping, shuffle)

    @torch.no_grad()
    def evaluate(self) -> RegressionEvalInfo | ClassificationEvalInfo:
        """Evaluate the model on the test set of the dataset.
        """
        if self.dataset.test is None and self.dataset.validation is not None:
            self.logger.warning("no test split found, using validation split for evaluation")
            dataset = self.dataset.validation
        elif self.dataset.test is not None:
            dataset = self.dataset.test
        else:
            raise ValueError("no test or validation split found")

        self.model.eval()
        self.likelihood.eval()

        if self.task == "regression":
            return self._evaluate_regression(dataset)
        else:
            return self._evaluate_classification(dataset)

    def save(self,
             path: str,
             dtype: str = "float32",
             ) -> None:
        """Save the model to a file.

        Arguments
        ---------
        """
        self._save(path, dtype)

    def _log_training_info(self) -> None:
        n_params: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        device_name: str = torch.cuda.get_device_name() if self.device.type == "cuda" else platform.processor()
        self.logger.info(f"""training ({self.task})
    -> model: {self.model.__class__.__name__} ({n_params} trainable parameters)
    -> device: {self.device} ({device_name})
        """)

    def _init(self,
              model: Module,
              dataset: CatastaDataset,
              optimizer: str,
              loss_function: str | Module,
              probabilistic: bool,
              device: str,
              dtype: str,
              ) -> None:
        self.task: str = dataset.task

        self.device: torch.device = _get_device(device)
        self.dtype: torch.dtype = _get_dtype(dtype)

        self.model: Module = model.to(self.device, self.dtype)
        self.likelihood: Module = GaussianLikelihood().to(self.device, self.dtype) if isinstance(self.model, GP) else Identity()

        self.dataset: CatastaDataset = dataset

        self.optimizer_id: str = optimizer
        self.loss_function_id: str | Module = loss_function

        self.logger: Logger = Logger("catasta")
        self._log_training_info()

    def _train(self,
               epochs: int,
               batch_size: int,
               lr: float,
               final_lr: float | None,
               early_stopping: tuple[int, float] | None,
               shuffle: bool,
               ) -> TrainInfo:
        # VARIABLES
        optimizer: Optimizer = get_optimizer(self.optimizer_id, [self.model, self.likelihood], lr)
        loss_function: Module = get_loss_function(self.loss_function_id, self.model, self.likelihood, len(self.dataset.train))  # type: ignore

        model_state_manager: ModelStateManager = ModelStateManager(early_stopping)
        lr_decay: float = (final_lr / lr) ** (1 / epochs) if final_lr else 1.0
        scheduler: StepLR = StepLR(optimizer, step_size=1, gamma=lr_decay)

        training_logger: TrainingLogger = TrainingLogger(epochs)

        # TRAINING LOOP
        for epoch in range(epochs):
            start_time: float = time.time()

            train_loss, train_accuracy = _train_epoch(self.task,
                                                      [self.model],
                                                      self.dataset.train,
                                                      shuffle,
                                                      batch_size,
                                                      optimizer,
                                                      loss_function,
                                                      self.device,
                                                      self.dtype,
                                                      )
            val_loss, val_accuracy = _val_epoch(self.task,
                                                [self.model, self.likelihood],
                                                self.dataset.validation,
                                                batch_size,
                                                loss_function,
                                                self.device,
                                                self.dtype,
                                                )

            scheduler.step()

            model_state_manager([self.model, self.likelihood], val_loss if val_loss is not None else train_loss)

            if model_state_manager.early_stop:
                self.logger.warning(f"early stopping at epoch {epoch + 1}")
                break

            time_per_epoch = time.time() - start_time

            training_logger.log(
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                lr=optimizer.param_groups[0]["lr"],
                epoch=epoch + 1,
                time_per_epoch=time_per_epoch,
            )

            self.logger.info(training_logger, flush=True)

        # END OF TRAINING
        train_info = training_logger.get_info()

        self.logger.info(f"training completed | best loss: {train_info.best_val_loss:.4f}")

        model_state_manager.load_best_model_state([self.model, self.likelihood])

        return train_info

    def _evaluate_regression(self, dataset: Dataset) -> RegressionEvalInfo:
        x, y = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))  # type: ignore

        x: Tensor = x.to(self.device, dtype=self.dtype)
        y: Tensor = y.to(self.device, dtype=self.dtype)

        output = self.likelihood(self.model(x))

        true_input: np.ndarray = x.cpu().numpy()[:, -1]
        true_output: np.ndarray = y.cpu().numpy()

        if isinstance(output, Distribution):
            predicted_output: np.ndarray = output.mean.cpu().numpy()
            predicted_output_std: np.ndarray = output.stddev.cpu().numpy()
        else:
            predicted_output: np.ndarray = output.cpu().numpy()
            predicted_output_std = np.zeros_like(true_output)

        return RegressionEvalInfo(true_input, true_output, predicted_output, predicted_output_std)

    def _evaluate_classification(self, dataset: Dataset) -> ClassificationEvalInfo:
        dataloader: DataLoader = DataLoader(dataset, batch_size=128, shuffle=False)

        true_labels: list[int] = []
        predicted_labels: list[int] = []
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device, dtype=self.dtype)
            y_batch = y_batch.to(self.device, dtype=torch.long)

            output: Tensor = self.model(x_batch)

            true_labels.append(y_batch.cpu().numpy())
            predicted_labels.append(output.argmax(dim=1).cpu().numpy())

        return ClassificationEvalInfo(
            true_labels=np.concatenate(true_labels),
            predicted_labels=np.concatenate(predicted_labels),
        )

    def _save(self, path: str, dtype: str) -> None:
        if "." in path:
            raise ValueError("save path must be a directory")

        if not os.path.exists(path):
            os.makedirs(path)

        model_dtype: torch.dtype = _get_dtype(dtype)
        model_device: torch.device = torch.device("cpu")

        self.model.to(model_device, model_dtype)
        self.likelihood.to(model_device, model_dtype)

        for model in [self.model, self.likelihood]:
            if isinstance(model, Identity):
                continue

            model_name: str = model.__class__.__name__
            model_path: str = os.path.join(path, f"{model_name}.pt")
            torch.save(model.state_dict(), model_path)


def _train_epoch(task: str,
                 models: list[Module],
                 dataset: Dataset,
                 shuffle: bool,
                 batch_size: int,
                 optimizer: Optimizer,
                 loss_function: Module,
                 device: torch.device,
                 dtype: torch.dtype,
                 ) -> tuple[float, float | None]:
    for model in models:
        model.train()

    data_loader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return _epoch(task, models, data_loader, optimizer, loss_function, device, dtype)


@torch.no_grad()
def _val_epoch(task: str,
               models: list[Module],
               dataset: Dataset | None,
               batch_size: int,
               loss_function: Module,
               device: torch.device,
               dtype: torch.dtype,
               ) -> tuple[float, float | None]:
    if dataset is None:
        return float("inf"), None

    for model in models:
        model.eval()

    data_loader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return _epoch(task, models, data_loader, None, loss_function, device, dtype)


def _epoch(task: str,
           models: list[Module],
           data_loader: DataLoader,
           optimizer: Optimizer | None,
           loss_function: Module,
           device: torch.device,
           dtype: torch.dtype,
           ) -> tuple[float, float | None]:
    cumulated_loss: float = 0.0
    total_samples: int = 0
    correct_predictions: int = 0
    for inputs, targets in data_loader:
        inputs: Tensor = inputs.to(device, dtype)
        targets: Tensor = targets.to(device, dtype if task == "regression" else torch.long)

        optimizer.zero_grad() if optimizer is not None else None

        outputs = models[0](inputs)
        if len(models) > 1:
            for model in models[1:]:
                outputs = model(outputs)

        loss: Tensor = loss_function(outputs, targets)
        if isinstance(outputs, Distribution):
            loss = -loss

        loss.backward() if optimizer is not None else None
        optimizer.step() if optimizer is not None else None

        cumulated_loss += loss.item() * inputs.shape[0]
        total_samples += inputs.shape[0]
        if task == "classification":
            correct_predictions += (outputs.argmax(dim=1) == targets).sum().item()  # type: ignore

    epoch_loss = cumulated_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples if task == "classification" else None

    return epoch_loss, epoch_accuracy
