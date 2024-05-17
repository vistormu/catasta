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
from gpytorch.mlls import MarginalLogLikelihood

from ..datasets import CatastaDataset
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
                 optimizer: str | Optimizer,
                 loss_function: str | Module,
                 probabilistic: bool = False,
                 likelihood: str | Module | None = None,
                 device: str = "auto",
                 dtype: str = "float32",
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
        probabilistic : bool, optional
            Wheter to use probabilistic training or not. Defaults to False. Not yet supported.
        likelihood : str | torch.nn.Module, optional
            The likelihood to be used for Gaussian Processes. If the user is training a GP model and the likelihood is not provided, the default Gaussian likelihood is used. One of: gaussian, bernoulli, laplace, softmax, studentt, beta. 
        device : str, optional
            The device to be used for training. One of "cpu", "cuda", or "auto". Defaults to "auto".
        dtype : str, optional
            The data type to be used for training. One of "float16", "float32", or "float64". Defaults to "float32".
        verbose : bool, optional
            Whether to print training information or not. Defaults to True.
        """
        self.task: str = dataset.task

        self.device: torch.device = _get_device(device)
        self.dtype: torch.dtype = _get_dtype(dtype)

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

        if probabilistic:
            self.logger.warning("probabilistic models are not yet supported")

    def _log_training_info(self) -> None:
        n_params: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        device_name: str = torch.cuda.get_device_name() if self.device.type == "cuda" else platform.processor()
        n_samples: int = len(self.dataset.train)  # type: ignore
        self.logger.info(f"""training
    -> task:     {self.task}
    -> dataset:  {self.dataset.root} ({n_samples} samples)
    -> model:    {self.model.__class__.__name__} ({n_params} trainable parameters)
    -> device:   {self.device} ({device_name})""")

    def train(self, *,
              epochs: int,
              batch_size: int,
              lr: float,
              final_lr: float | None = None,
              early_stopping: bool = False,
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
        final_lr : float, optional
            The final learning rate. If not provided, the learning rate is not decayed. Defaults to None.
        early_stopping : bool, optional
            Whether to use early stopping or not. The criterion for early stopping is the derivative of the validation loss. Defaults to False.
        shuffle : bool, optional
            Whether to shuffle the training data or not. Defaults to True.
        data_loader_workers : int, optional
            The number of workers to use for the data loader. Defaults to 0.
        """
        # VARIABLES
        optimizer: Optimizer = get_optimizer(self.optimizer_id, [self.model, self.likelihood], lr)
        loss_function: Module = get_loss_function(self.loss_function_id, self.model, self.likelihood, len(self.dataset.train))  # type: ignore

        model_state_manager: ModelStateManager = ModelStateManager(early_stopping)
        lr_decay: float = (final_lr / lr) ** (1 / epochs) if final_lr else 1.0
        scheduler: StepLR = StepLR(optimizer, step_size=1, gamma=lr_decay)

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

        # TRAINING LOOP
        for epoch in range(epochs):
            start_time: float = time.time()

            # train
            self.model.train()
            self.likelihood.train()

            train_loss, train_accuracy = self._epoch(train_data_loader, optimizer, loss_function)

            # validation
            self.model.eval()
            self.likelihood.eval()

            with torch.no_grad():
                val_loss, val_accuracy = self._epoch(val_data_loader, None, loss_function)

            scheduler.step()

            torch.cuda.empty_cache() if self.device.type == "cuda" else None

            model_state_manager([self.model, self.likelihood], val_loss)

            if model_state_manager.stop:
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
             dtype: str = "float32",
             ) -> None:
        """Save the model to a file.

        Arguments
        ---------
        path : str
            The directory to save the model to.
        dtype : str, optional
            The data type to save the model in. Can be "float16", "float32", or "float64". Defaults to "float32".

        Raises
        ------
        ValueError
            If the path is a file path.
        """
        if "." in path:
            raise ValueError("save path must be a directory")

        model_dtype: torch.dtype = _get_dtype(dtype)
        model_device: torch.device = torch.device("cpu")

        save_path: str = os.path.join(path, self.model.__class__.__name__)

        os.makedirs(save_path, exist_ok=True)

        self.model.to(model_device, model_dtype)
        self.likelihood.to(model_device, model_dtype)

        for model in [self.model, self.likelihood]:
            if isinstance(model, Identity):
                continue

            model_path: str = os.path.join(save_path, f"{model.__class__.__name__}.pt")
            torch.save(model, model_path)

            self.logger.info(f"model saved to {model_path}")

    @torch.no_grad()
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

    @torch.no_grad()
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
               optimizer: Optimizer | None,
               loss_function: Module,
               ) -> tuple[float, float]:
        cumulated_loss: float = 0.0
        total_samples: int = 0
        correct_predictions: int = 0
        for inputs, targets in data_loader:
            inputs: Tensor = inputs.to(self.device, self.dtype)
            targets: Tensor = targets.to(self.device, self.dtype if self.task == "regression" else torch.long)

            optimizer.zero_grad() if optimizer is not None else None

            output = self.likelihood(self.model(inputs))

            loss: Tensor = loss_function(output, targets)  # type: ignore
            if isinstance(loss_function, MarginalLogLikelihood):
                loss = -loss

            loss.backward() if optimizer is not None else None
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) if optimizer is not None else None  # type: ignore

            optimizer.step() if optimizer is not None else None

            cumulated_loss += loss.item() * inputs.shape[0]
            total_samples += inputs.shape[0]
            if self.task == "classification":
                correct_predictions += (output.argmax(dim=1) == targets).sum().item()  # type: ignore

        epoch_loss = cumulated_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples if self.task == "classification" else -np.inf

        return epoch_loss, epoch_accuracy
