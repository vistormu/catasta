import platform
import time
import os
import gc

import numpy as np

import torch
from torch import Tensor
from torch.nn import Module, Identity
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torch.distributions import Distribution

from gpytorch.models.gp import GP
from gpytorch.mlls import MarginalLogLikelihood

from ..dataset import CatastaDataset
from .utils import (
    get_optimizer,
    get_loss_function,
    get_likelihood,
    ModelStateManager,
    TrainingLogger,
)
from ..dataclasses import TrainInfo, EvalInfo

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


class Scaffold:
    def __init__(self, *,
                 model: Module,
                 dataset: CatastaDataset,
                 optimizer: str | Optimizer,
                 loss_function: str | Module,
                 likelihood: str | Module | None = None,
                 device: str = "auto",
                 verbose: bool = True,
                 ) -> None:
        """The `Scaffold` class is a high-level API for training models on Catasta datasets.

        Arguments
        ---------
        model : torch.nn.Module
            The model to be trained.
        dataset : ~catasta.datasets.CatastaDataset
            The dataset to be used for training, validation, and testing.
        optimizer : str | torch.optim.Optimizer
            The optimizer to be used for training. The user can either pass a string with the name of the optimizer or a custom optimizer. One of: adam, sgd, adamw, lbfgs, rmsprop, rprop, adadelta, adagrad, adamax, asgd, sparseadam.
        loss_function : str | torch.nn.Module
            The loss function to be used for training. The user can either pass a string with the name of the loss function or a custom loss function. A list of available loss functions is in ~catasta.scaffolds.utils.available_loss_functions
        likelihood : str | torch.nn.Module, optional
            The likelihood to be used for Gaussian Processes. If the user is training a GP model and the likelihood is not provided, the default Gaussian likelihood is used. One of: gaussian, bernoulli, laplace, softmax, studentt, beta.
        device : str, optional
            The device to be used for training. One of "cpu", "cuda", or "auto". Defaults to "auto".
        verbose : bool, optional
            Whether to print training information or not. Defaults to True.
        """
        self.task: str = dataset.task

        self.device: torch.device = _get_device(device)
        self.dtype: torch.dtype = torch.float32

        self.verbose: bool = verbose

        self.model: Module = model.to(self.device, self.dtype)
        if isinstance(self.model, GP) and likelihood is None:
            self.likelihood: Module = get_likelihood("gaussian").to(self.device, self.dtype)
        else:
            self.likelihood: Module = get_likelihood(likelihood).to(self.device, self.dtype)

        self.dataset: CatastaDataset = dataset

        self.optimizer_id: str | Optimizer = optimizer
        self.loss_function_id: str | Module = loss_function

        self.logger: Logger = Logger("catasta", disable=not verbose)
        self._log_training_info()

        torch.backends.cudnn.benchmark = True

    def _log_training_info(self) -> None:
        n_params: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        device_name: str = torch.cuda.get_device_name() if self.device.type == "cuda" else platform.processor()
        n_samples: int = len(self.dataset.train)  # type: ignore
        self.logger.info(f"""   TRAINING INFO
    -> task:     {self.task}
    -> dataset:  {self.dataset.root} ({n_samples} samples)
    -> model:    {self.model.__class__.__name__} ({n_params} trainable parameters)
    -> device:   {self.device} ({device_name})""")

    def train(self, *,
              epochs: int,
              batch_size: int,
              lr: float,
              max_lr: float | None = None,
              early_stopping_alpha: float | None = None,
              shuffle: bool = True,
              data_loader_workers: int = 0,
              ) -> TrainInfo:
        """Train the model on the dataset.

        Arguments
        ---------
        epochs : int
            The number of epochs to train the model.
        batch_size : int
            The batch size to use for training.
        lr : float
            The initial learning rate.
        max_lr : float, optional
            The maximum learning rate to use for the OneCycleLR scheduler. If not provided, the learning rate will not decay. Defaults to None.
        early_stopping_alpha : bool, optional
            The smoothing factor for the early stopping criterion. If set to a value, the training will stop when the validation loss curve starts increasing. A good smoothing factor ranges in [0.95, 1). Defaults to None.
        shuffle : bool, optional
            Whether to shuffle the training data or not. Defaults to True.
        data_loader_workers : int, optional
            The number of workers to use for the data loader. Defaults to 0. Tip: set to 4 times the number of GPUs.
        """
        # VARIABLES
        optimizer: Optimizer = get_optimizer(self.optimizer_id, [self.model, self.likelihood], lr)
        loss_function: Module = get_loss_function(self.loss_function_id, self.model, self.likelihood, len(self.dataset.train))  # type: ignore

        model_state_manager: ModelStateManager = ModelStateManager(early_stopping_alpha)

        training_logger: TrainingLogger = TrainingLogger(self.task, epochs)

        train_data_loader: DataLoader = DataLoader(
            self.dataset.train,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=data_loader_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )
        val_data_loader: DataLoader = DataLoader(
            self.dataset.validation,
            batch_size=batch_size,
            shuffle=False,
            num_workers=data_loader_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        scheduler: OneCycleLR | None = OneCycleLR(optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(train_data_loader)) if max_lr else None

        gc.collect()
        torch.cuda.empty_cache()

        # TRAINING LOOP
        for epoch in range(epochs):
            start_time: float = time.time()

            # train
            self.model.train()
            self.likelihood.train()
            train_loss, train_accuracy = self._epoch(train_data_loader, loss_function, optimizer, scheduler)

            # validation
            self.model.eval()
            self.likelihood.eval()
            with torch.no_grad():
                val_loss, val_accuracy = self._epoch(val_data_loader, loss_function, None, None)

            model_state_manager([self.model, self.likelihood], val_loss)

            if model_state_manager.stop:
                self.logger.warning(f"early stopping at epoch {epoch + 1}")
                break

            training_logger.log(
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                lr=optimizer.param_groups[0]["lr"],
                epoch=epoch + 1,
                time_per_epoch=time.time() - start_time,
            )

            Logger.plain(training_logger, color="green") if self.verbose else None

        # END OF TRAINING
        train_info = training_logger.get_info()

        model_state_manager.load_best_model_state([self.model, self.likelihood])

        return train_info

    def evaluate(self) -> EvalInfo:
        """Evaluate the model on the test set of the dataset.

        Returns
        -------
        ~catasta.dataclasses.EvalInfo
            The evaluation information.
        """
        self.model.eval()
        self.likelihood.eval()

        if self.task == "regression":
            return self._evaluate_regression(self.dataset.test)
        else:
            return self._evaluate_classification(self.dataset.test)

    def save(self,
             path: str,
             ) -> None:
        """Save the model to a file.

        Arguments
        ---------
        path : str
            The directory to save the model to.

        Raises
        ------
        ValueError
            If the path is a file path.
        """
        if "." in path:
            raise ValueError("save path must be a directory")

        model_device: torch.device = torch.device("cpu")

        save_path: str = os.path.join(path, self.model.__class__.__name__)

        os.makedirs(save_path, exist_ok=True)

        self.model.to(model_device)
        self.likelihood.to(model_device)

        for model in [self.model, self.likelihood]:
            if isinstance(model, Identity):
                continue

            model_path: str = os.path.join(save_path, f"{model.__class__.__name__}.pt")
            torch.save(model, model_path)

            self.logger.info(f"model saved to {model_path}")

    @ torch.no_grad()
    def _evaluate_regression(self, dataset: Dataset) -> EvalInfo:
        x, y = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))  # type: ignore

        x: Tensor = x.to(self.device, dtype=self.dtype)
        y: Tensor = y.to(self.device, dtype=self.dtype)

        output = self.likelihood(self.model(x))

        true_output: np.ndarray = y.cpu().numpy()

        if isinstance(output, Distribution):
            predicted_output: np.ndarray = output.mean.cpu().numpy()
            predicted_output_std: np.ndarray = output.stddev.cpu().numpy()
        else:
            predicted_output: np.ndarray = output.cpu().numpy()
            predicted_output_std = np.zeros_like(true_output)

        return EvalInfo(
            task="regression",
            true_output=true_output,
            predicted_output=predicted_output,
            predicted_std=predicted_output_std,
        )

    @ torch.no_grad()
    def _evaluate_classification(self, dataset: Dataset) -> EvalInfo:
        dataloader: DataLoader = DataLoader(dataset, batch_size=128, shuffle=False)

        true_labels: list[int] = []
        predicted_labels: list[int] = []
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device, dtype=self.dtype)
            y_batch = y_batch.to(self.device, dtype=torch.long)

            output: Tensor = self.model(x_batch)

            true_labels.append(y_batch.cpu().numpy())
            predicted_labels.append(output.argmax(dim=1).cpu().numpy())

        return EvalInfo(
            task="classification",
            true_output=np.concatenate(true_labels),
            predicted_output=np.concatenate(predicted_labels),
        )

    def _epoch(self,
               data_loader: DataLoader,
               loss_function: Module,
               optimizer: Optimizer | None,
               scheduler: OneCycleLR | None,
               ) -> tuple[float, float]:
        cumulated_loss: float = 0.0
        total_samples: int = 0
        correct_predictions: int = 0
        for inputs, targets in data_loader:
            inputs: Tensor = inputs.to(self.device, self.dtype, non_blocking=True)
            targets: Tensor = targets.to(self.device, self.dtype if self.task == "regression" else torch.long, non_blocking=True)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            output = self.likelihood(self.model(inputs))

            loss: Tensor = loss_function(output, targets)  # type: ignore
            if isinstance(loss_function, MarginalLogLikelihood):
                loss = -loss

            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # type: ignore
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            cumulated_loss += loss.item() * inputs.shape[0]
            total_samples += inputs.shape[0]

            if self.task == "classification":
                correct_predictions += (output.argmax(dim=1) == targets).sum().item()  # type: ignore

        epoch_loss = cumulated_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples if self.task == "classification" else -np.inf

        return epoch_loss, epoch_accuracy
