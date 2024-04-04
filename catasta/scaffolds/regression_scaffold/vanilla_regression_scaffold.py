import time
import os

import numpy as np

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.nn.modules.loss import _Loss
import torch.onnx as onnx

from vclog import Logger

from .regression_scaffold_interface import RegressionScaffold
from ...datasets import RegressionDataset
from ...dataclasses import RegressionEvalInfo, RegressionTrainInfo
from ...utils import get_optimizer, get_loss_function, RegressionTrainingLogger, ModelStateManager


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
        self.context_length: int = 0

        self.dataset: RegressionDataset = dataset

        self.optimmizer_id: str = optimizer
        self.loss_function_id: str = loss_function

        self.logger: Logger = Logger("catasta")

        # Logging info
        message: str = f"using {self.device} with {torch.cuda.get_device_name()}" if torch.cuda.is_available() else f"using {self.device}"
        self.logger.info(message)
        n_parameters: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"training model {self.model.__class__.__name__} ({n_parameters} parameters)")

    def train(self, *,
              epochs: int = 100,
              batch_size: int = 128,
              lr: float = 1e-3,
              final_lr: float | None = None,
              early_stopping: tuple[int, float] | None = None,
              ) -> RegressionTrainInfo:
        self.model.train()

        if self.dataset.validation is None:
            self.logger.warning("no validation split found")

        optimizer: Optimizer = get_optimizer(self.optimmizer_id, self.model, lr)
        loss_function: _Loss = get_loss_function(self.loss_function_id)

        lr_decay: float = (final_lr / lr) ** (1 / epochs) if final_lr is not None else 1.0
        scheduler: StepLR = StepLR(optimizer, step_size=1, gamma=lr_decay)

        model_state_manager = ModelStateManager(early_stopping)

        training_logger = RegressionTrainingLogger(epochs)

        data_loader: DataLoader = DataLoader(self.dataset.train, batch_size=batch_size, shuffle=True)

        time_per_epoch: float = 0.0
        for epoch in range(epochs):
            batch_train_losses: list[float] = []
            start_time: float = time.time()
            for x_batch, y_batch in data_loader:
                optimizer.zero_grad()

                if not self.context_length:
                    self.context_length = x_batch.shape[1]

                x_batch = x_batch.to(self.device, dtype=self.dtype)
                y_batch = y_batch.to(self.device, dtype=self.dtype)

                output: Tensor = self.model(x_batch)

                loss: Tensor = loss_function(output, y_batch)
                loss.backward()

                optimizer.step()

                batch_train_losses.append(loss.item())

            # END OF EPOCH
            scheduler.step()

            val_loss = self._estimate_loss(batch_size)

            model_state_manager(self.model.state_dict(), val_loss)

            if model_state_manager.stop():
                self.logger.warning("early stopping")
                break

            time_per_epoch = time.time() - start_time

            training_logger.log(
                train_loss=np.mean(batch_train_losses).astype(float),
                val_loss=val_loss,
                lr=scheduler.get_last_lr()[0],
                epoch=epoch + 1,
                time_per_epoch=time_per_epoch,
            )

            self.logger.info(training_logger, flush=True)

        # END OF TRAINING
        train_info: RegressionTrainInfo = training_logger.get_regression_train_info()

        self.logger.info(f'training completed | best loss: {train_info.best_val_loss:.4f}')

        model_state_manager.load_best_model_state(self.model)

        return train_info

    @torch.no_grad()
    def evaluate(self) -> RegressionEvalInfo:
        if self.dataset.test is None and self.dataset.validation is not None:
            self.logger.warning("no test split found, using validation split for evaluation")
            self.dataset.test = self.dataset.validation
        else:
            raise ValueError(f"cannot evaluate without a test split")

        self.model.eval()

        x, y = next(iter(DataLoader(self.dataset.test, batch_size=len(self.dataset.test), shuffle=False)))

        x: Tensor = x.to(self.device, dtype=self.dtype)
        y: Tensor = y.to(self.device, dtype=self.dtype)

        output: Tensor = self.model(x)

        true_input: np.ndarray = x.cpu().numpy()[:, -1]
        true_output: np.ndarray = y.cpu().numpy()
        predicted_output: np.ndarray = output.cpu().numpy()

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
            x_batch = x_batch.to(self.device, dtype=self.dtype)
            y_batch = y_batch.to(self.device, dtype=self.dtype)

            output: Tensor = self.model(x_batch)

            loss: Tensor = loss_function(output, y_batch)

            losses.append(loss.item())

        self.model.train()

        return np.mean(losses).astype(float)

    def save(self, *,
             path: str,
             to_onnx: bool = False,
             dtype: str = "float32",
             context_length: int | None = None,
             ) -> None:
        if "." in path:
            raise ValueError("save path must be a directory")

        if not os.path.exists(path):
            os.makedirs(path)

        if dtype not in ["float16", "float32", "float64"]:
            raise ValueError(f"invalid dtype: {dtype}")

        model_dtype = torch.float16 if dtype == "float16" else torch.float32 if dtype == "float32" else torch.float64
        model_device = torch.device("cpu")
        self.model = self.model.to(model_device, model_dtype)

        model_name: str = self.model.__class__.__name__

        if not to_onnx:
            model_path = os.path.join(path, f"{model_name}.pt")
            torch.save(self.model.state_dict(), model_path)
        else:
            if not self.context_length and not context_length:
                raise ValueError("could not infer the context length for the model. Please, provide it manually.")

            context_length = self.context_length if not context_length else context_length
            dummy_input = torch.randn(1, context_length).to(model_dtype)
            model_path = os.path.join(path, f"{model_name}.onnx")

            onnx.export(
                self.model,
                dummy_input,
                model_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size", 1: "context_length"},
                              "output": {0: "prediction"}},
            )

        self.logger.info(f"saved model {model_name} to {path}")
