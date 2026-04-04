import gc
import os
import platform
import time
from typing import Any, Callable

import numpy as np

import torch
from gpytorch.mlls import MarginalLogLikelihood
from torch import Tensor
from torch.distributions import Distribution
from torch.nn import Module
from torch.optim import LBFGS, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ..dataclasses import EvalInfo, TrainInfo
from ..dataset import CatastaDataset
from ..log import ansi
from .utils import ModelStateManager, TrainingLogger, get_loss_function, get_optimizer

SchedulerType = Any
SchedulerConfig = SchedulerType | tuple[SchedulerType, str]
EarlyStoppingConfig = tuple[str, float | int, float | int] | None


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


def _normalize_scheduler(scheduler: SchedulerConfig | None) -> tuple[SchedulerType | None, str]:
    if scheduler is None:
        return None, "epoch"

    if isinstance(scheduler, tuple):
        if len(scheduler) != 2 or not isinstance(scheduler[1], str):
            raise ValueError("scheduler tuple must be (scheduler_instance, step_mode)")
        return scheduler[0], scheduler[1]

    return scheduler, "epoch"


class Scaffold:
    def __init__(
        self,
        *,
        model: Module,
        dataset: CatastaDataset,
        optimizer: str | Optimizer,
        loss_function: str | Module,
        device: str = "auto",
        verbose: bool = True,
    ) -> None:
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

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        device_name = torch.cuda.get_device_name() if self.device.type == "cuda" else platform.processor()
        n_samples = len(self.dataset.train)  # type: ignore[arg-type]
        print(
            f"{ansi.BOLD}{ansi.GREEN}-> training info\n{ansi.RESET}"
            f"   |> task:    {self.task}\n"
            f"   |> dataset: {self.dataset.root} ({n_samples} samples)\n"
            f"   |> model:   {self.model.__class__.__name__} ({n_params} parameters)\n"
            f"   |> device:  {self.device} ({device_name})"
        )

    def train(
        self,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        scheduler: SchedulerConfig | None = None,
        early_stopping: EarlyStoppingConfig = None,
        shuffle: bool = True,
        data_loader_workers: int = 0,
    ) -> TrainInfo:
        optimizer: Optimizer = get_optimizer(self.optimizer_id, self.model, lr)
        loss_function: Module = get_loss_function(self.loss_function_id, self.model, len(self.dataset.train))  # type: ignore[arg-type]

        scheduler_obj, scheduler_step_mode = _normalize_scheduler(scheduler)
        if scheduler_step_mode == "val_loss" and not isinstance(scheduler_obj, ReduceLROnPlateau):
            raise ValueError("scheduler mode 'val_loss' requires torch.optim.lr_scheduler.ReduceLROnPlateau")

        model_state_manager = ModelStateManager("none" if early_stopping is None else early_stopping)
        training_logger = TrainingLogger(self.task, epochs)

        train_data_loader = DataLoader(
            self.dataset.train,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=data_loader_workers,
            pin_memory=self.device.type == "cuda",
        )
        val_data_loader = DataLoader(
            self.dataset.validation,
            batch_size=batch_size,
            shuffle=False,
            num_workers=data_loader_workers,
            pin_memory=self.device.type == "cuda",
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for epoch in range(epochs):
            try:
                start_time = time.time()

                self.model.train()
                train_loss = self._epoch(train_data_loader, loss_function, optimizer)

                self.model.eval()
                with torch.no_grad():
                    val_loss = self._epoch(val_data_loader, loss_function, None)

                model_state_manager(self.model, val_loss)

                training_logger.log(
                    train_loss=train_loss,
                    val_loss=val_loss,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    epoch=epoch + 1,
                    time_per_epoch=time.time() - start_time,
                )

                self._step_scheduler(scheduler_obj, scheduler_step_mode, train_loss, val_loss)

                if self.verbose:
                    print(training_logger)

                if model_state_manager.stop:
                    if self.verbose:
                        print(
                            f"\n{ansi.BOLD}{ansi.YELLOW}-> early stopping{ansi.RESET}\n"
                            f"   |> epoch: {epoch + 1}"
                        )
                    break

            except KeyboardInterrupt:
                if self.verbose:
                    print(
                        f"\n{ansi.BOLD}{ansi.YELLOW}-> training interrupted{ansi.RESET}\n"
                        f"   |> epoch: {epoch + 1}"
                    )
                break

        train_info = training_logger.get_info()
        model_state_manager.load_best_model_state(self.model)
        return train_info

    def _step_scheduler(
        self,
        scheduler: SchedulerType | None,
        step_mode: str,
        train_loss: float,
        val_loss: float,
    ) -> None:
        if scheduler is None:
            return

        if step_mode == "epoch":
            scheduler.step()
            return

        if step_mode == "train_loss":
            scheduler.step(train_loss)
            return

        if step_mode == "val_loss":
            scheduler.step(val_loss)
            return

        raise ValueError("scheduler step mode must be one of: 'epoch', 'train_loss', 'val_loss'")

    def _epoch(
        self,
        data_loader: DataLoader,
        loss_function: Module,
        optimizer: Optimizer | None,
    ) -> float:
        cumulated_loss = torch.tensor(0.0, device=self.device)
        total_samples = 0

        for inputs, targets in data_loader:
            inputs = inputs.to(self.device, self.input_dtype, non_blocking=True)
            targets = targets.to(self.device, self.output_dtype, non_blocking=True)

            if optimizer is not None and isinstance(optimizer, LBFGS):
                def closure() -> Tensor:
                    optimizer.zero_grad(set_to_none=True)
                    output = self.model(inputs)
                    loss = loss_function(output, targets)  # type: ignore[arg-type]
                    if isinstance(loss_function, MarginalLogLikelihood):
                        loss = -loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # type: ignore[arg-type]
                    return loss

                loss = optimizer.step(closure)
                batch_loss = loss.detach()
            else:
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

                output = self.model(inputs)
                loss = loss_function(output, targets)  # type: ignore[arg-type]
                if isinstance(loss_function, MarginalLogLikelihood):
                    loss = -loss

                if optimizer is not None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # type: ignore[arg-type]
                    optimizer.step()

                batch_loss = loss.detach()

            cumulated_loss += batch_loss * inputs.shape[0]
            total_samples += inputs.shape[0]

        if total_samples == 0:
            return 0.0

        return float((cumulated_loss / total_samples).item())

    @torch.no_grad()
    def evaluate(
        self,
        batch_size: int | None = None,
        prediction_function: Callable[[Module, Any], Any] | None = None,
    ) -> EvalInfo:
        self.model.eval()

        effective_batch_size = len(self.dataset.test) if batch_size is None else batch_size  # type: ignore[arg-type]
        data_loader = DataLoader(
            self.dataset.test,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

        true_input: list[np.ndarray] = []
        true_output: list[np.ndarray] = []
        predicted_output: list[np.ndarray] = []
        for x, y in data_loader:
            x = x.to(self.device, self.input_dtype)
            output = self.model(x) if prediction_function is None else prediction_function(self.model, x)

            if isinstance(output, Distribution):
                output = output.mean

            if isinstance(output, Tensor):
                output_np = output.detach().cpu().numpy()
            else:
                output_np = np.asarray(output)

            true_input.append(x.detach().cpu().numpy())
            true_output.append(y.detach().cpu().numpy())
            predicted_output.append(output_np)

        return EvalInfo(
            task=self.task,
            true_input=np.concatenate(true_input),
            true_output=np.concatenate(true_output),
            predicted_output=np.concatenate(predicted_output),
        )

    def save(self, path: str) -> None:
        is_model_file = path.endswith((".pt", ".pth", ".onnx"))
        if is_model_file:
            parent_dir = os.path.dirname(path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
                print(
                    f"{ansi.BOLD}{ansi.YELLOW}-> specified path does not exist{ansi.RESET}\n"
                    f"   |> created directory: {parent_dir}"
                )
        elif not os.path.exists(path):
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
            try:
                import onnx  # type: ignore # noqa: F401
            except ImportError:
                print(
                    f"{ansi.BOLD}{ansi.RED}-> onnx library not found{ansi.RESET}\n"
                    f"   |> onnx is a special use case, please install it manually\n"
                    f"   |> the model will be saved as a `.pt` file\n"
                    f"   |> run: pip install onnx"
                )
                fallback_path = path.replace(".onnx", ".pt")
                torch.save(self.model, fallback_path)
                print(
                    f"{ansi.BOLD}{ansi.GREEN}-> model saved{ansi.RESET}\n"
                    f"   |> path: {fallback_path}"
                )
                return

            input_shape = self.dataset.train[0][0].shape
            dummy_input = torch.randn(1, *input_shape, dtype=self.input_dtype)
            torch.onnx.export(
                self.model,
                dummy_input,  # type: ignore[arg-type]
                path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
        else:
            raise ValueError("Invalid path. Must be a directory, a `.pt` or `.onnx` file.")

        print(
            f"{ansi.BOLD}{ansi.GREEN}-> model saved{ansi.RESET}\n"
            f"   |> path: {path}"
        )
