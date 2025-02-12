import platform
import time
import os
import gc

import numpy as np

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.distributions import Distribution

from gpytorch.mlls import MarginalLogLikelihood

from ..dataset import CatastaDataset
from .utils import (
    get_optimizer,
    get_loss_function,
    ModelStateManager,
    TrainingLogger,
)
from ..dataclasses import TrainInfo, EvalInfo
from ..log import ansi


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
        device : str, optional
            The device to be used for training. One of "cpu", "cuda", or "auto". Defaults to "auto".
        verbose : bool, optional
            Whether to print training information or not. Defaults to True.
        """
        self.task: str = dataset.task

        self.device: torch.device = _get_device(device)
        self.input_dtype: torch.dtype = torch.float32
        self.output_dtype: torch.dtype = torch.float32 if self.task == "regression" else torch.long

        self.verbose: bool = verbose

        self.model: Module = model.to(self.device, self.input_dtype)

        self.dataset: CatastaDataset = dataset

        self.optimizer_id: str | Optimizer = optimizer
        self.loss_function_id: str | Module = loss_function

        if not self.verbose:
            return

        n_params: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        device_name: str = torch.cuda.get_device_name() if self.device.type == "cuda" else platform.processor()
        n_samples: int = len(self.dataset.train)  # type: ignore
        print(
            f"{ansi.BOLD}{ansi.GREEN}-> training info\n{ansi.RESET}"
            f"   |> task:    {self.task}\n"
            f"   |> dataset: {self.dataset.root} ({n_samples} samples)\n"
            f"   |> model:   {self.model.__class__.__name__} ({n_params} parameters)\n"
            f"   |> device:  {self.device} ({device_name})"
        )

    def train(self, *,
              epochs: int,
              batch_size: int,
              lr: float,
              early_stopping_alpha: float | None = None,
              shuffle: bool = True,
              data_loader_workers: int = 0,
              ) -> TrainInfo:
        """Train the model on the dataset.

        Arguments
        ---------
        epochs: int
            The number of epochs to train the model.
        batch_size: int
            The batch size to use for training.
        lr: float
            The initial learning rate.
        early_stopping_alpha: bool, optional
            The smoothing factor for the early stopping criterion. If set to a value, the training will stop when the validation loss curve starts increasing. A good smoothing factor ranges in [0.95, 1). Defaults to None.
        shuffle: bool, optional
            Whether to shuffle the training data or not . Defaults to True.
        data_loader_workers: int, optional
            The number of workers to use for the data loader. Defaults to 0. Tip: set to 4 times the number of GPUs.
        """
        # VARIABLES
        optimizer: Optimizer = get_optimizer(self.optimizer_id, self.model, lr)
        loss_function: Module = get_loss_function(self.loss_function_id, self.model, len(self.dataset.train))  # type: ignore

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

        gc.collect()
        torch.cuda.empty_cache()

        # TRAINING LOOP
        for epoch in range(epochs):
            try:
                start_time: float = time.time()

                # train
                self.model.train()
                train_loss = self._epoch(train_data_loader, loss_function, optimizer)

                # validation
                self.model.eval()
                with torch.no_grad():
                    val_loss = self._epoch(val_data_loader, loss_function, None)

                model_state_manager(self.model, val_loss)

                if model_state_manager.stop:
                    print(
                        f"\n{ansi.BOLD}{ansi.YELLOW}-> early stopping{ansi.RESET}\n"
                        f"   |> epoch: {epoch + 1}"
                    )
                    break

                training_logger.log(
                    train_loss=train_loss,
                    val_loss=val_loss,
                    lr=optimizer.param_groups[0]["lr"],
                    epoch=epoch + 1,
                    time_per_epoch=time.time() - start_time,
                )

                print(training_logger) if self.verbose else None

            except KeyboardInterrupt:
                print(
                    f"\n{ansi.BOLD}{ansi.YELLOW}-> training interrupted{ansi.RESET}\n"
                    f"   |> epoch: {epoch + 1}"
                )
                break

        # END OF TRAINING
        train_info = training_logger.get_info()

        model_state_manager.load_best_model_state(self.model)

        return train_info

    def _epoch(self,
               data_loader: DataLoader,
               loss_function: Module,
               optimizer: Optimizer | None,
               ) -> float:
        cumulated_loss: Tensor = torch.tensor(0.0, device=self.device)
        total_samples: int = 0

        for inputs, targets in data_loader:
            inputs = inputs.to(self.device, self.input_dtype, non_blocking=True)
            targets = targets.to(self.device, self.output_dtype, non_blocking=True)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            output = self.model(inputs)

            loss: Tensor = loss_function(output, targets)  # type: ignore
            if isinstance(loss_function, MarginalLogLikelihood):
                loss = -loss

            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # type: ignore
                optimizer.step()

            cumulated_loss += loss.detach() * inputs.shape[0]
            total_samples += inputs.shape[0]

        epoch_loss = cumulated_loss / total_samples if total_samples > 0 else torch.tensor(0.0, device=self.device)

        return epoch_loss.item()

    @torch.no_grad()
    def evaluate(self, batch_size: int | None = None) -> EvalInfo:
        """Evaluate the model on the test set of the dataset.

        Arguments
        ---------
        batch_size: int, optional
            The batch size to use for evaluation. If None, it uses the entire test set. Defaults to None.

        Returns
        -------
        ~catasta.dataclasses.EvalInfo
            The evaluation information.
        """
        self.model.eval()

        batch_size = len(self.dataset.test) if batch_size is None else batch_size  # type: ignore
        data_loader: DataLoader = DataLoader(
            self.dataset.test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        true_output = []
        predicted_output = []
        predicted_output_std = []
        for x, y in data_loader:
            x = x.to(self.device, self.input_dtype)

            output = self.model(x)
            true_output.append(y.cpu().numpy())

            if isinstance(output, Distribution):
                predicted_output.append(output.mean.cpu().numpy())
                predicted_output_std.append(output.stddev.cpu().numpy())
            else:
                predicted_output.append(output.cpu().numpy())
                predicted_output_std.append(np.zeros_like(output.cpu().numpy()))

        return EvalInfo(
            task=self.task,
            true_output=np.concatenate(true_output),
            predicted_output=np.concatenate(predicted_output),
            predicted_std=np.concatenate(predicted_output_std),
        )

    def save(self, path: str) -> None:
        """Save the model to a file.

        Arguments
        ---------
        path : str
            The path where to save the model. If the path is a directory, the model is saved as the model class name in the directory. If the path ends with `.onnx`, the model is saved in ONNX format.

        Raises
        ------
        ValueError
            If the path is not a directory, a `.pt` or `.onnx` file.
        """

        if "." in path and not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
            print(
                f"{ansi.BOLD}{ansi.YELLOW}-> specified path does not exist{ansi.RESET}\n"
                f"   |> created directory: {os.path.dirname(path)}"
            )

        if "." not in path and not os.path.exists(path):
            os.makedirs(path)
            print(
                f"{ansi.BOLD}{ansi.YELLOW}-> specified path does not exist{ansi.RESET}\n"
                f"   |> created directory: {path}"
            )

        self.model.to("cpu")

        if os.path.isdir(path):
            path = os.path.join(path, f"{self.model.__class__.__name__}.pt")

        if path.endswith(".pt") or path.endswith(".pth"):
            torch.save(self.model, path)

        elif path.endswith(".onnx"):
            # check if the onnx library is installed
            try:
                import onnx  # type: ignore
            except ImportError:
                print(
                    f"{ansi.BOLD}{ansi.RED}-> onnx library not found{ansi.RESET}\n"
                    f"   |> onnx is a special use case, please install it manually\n"
                    f"   |> the model will be saved as a `.pt` file\n"
                    f"   |> run: pip install onnx"
                )

                torch.save(self.model, path.replace(".onnx", ".pt"))

                print(
                    f"{ansi.BOLD}{ansi.GREEN}-> model saved{ansi.RESET}\n"
                    f"   |> path: {path.replace('.onnx', '.pt')}"
                )
                return

            input_shape = self.dataset.train[0][0].shape
            dummy_input = torch.randn(1, *input_shape, dtype=self.input_dtype)

            torch.onnx.export(
                self.model,
                dummy_input,  # type: ignore
                path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"}
                }
            )
        else:
            raise ValueError("Invalid path. Must be a directory, a `.pt` or `.onnx` file.")

        print(
            f"{ansi.BOLD}{ansi.GREEN}-> model saved{ansi.RESET}\n"
            f"   |> path: {path}"
        )
