import time
import os

import numpy as np

import torch
import torch.onnx as onnx
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.nn.modules.loss import _Loss

from vclog import Logger

from ...datasets import ClassificationDataset
from ...dataclasses import ClassificationTrainInfo, ClassificationEvalInfo
from ...utils import get_optimizer, get_loss_function, ClassificationTrainingLogger, ModelStateManager


class VanillaClassificationScaffold:
    def __init__(self, *,
                 model: Module,
                 dataset: ClassificationDataset,
                 optimizer: str = "adam",
                 loss_function: str = "mse",
                 ) -> None:
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype: torch.dtype = torch.float32

        self.model: Module = model.to(self.device)
        self.shape: tuple | None = None

        self.dataset: ClassificationDataset = dataset

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
              ) -> ClassificationTrainInfo:
        self.model.train()

        if self.dataset.validation is None:
            self.logger.warning("no validation split found")

        optimizer: Optimizer = get_optimizer(self.optimmizer_id, self.model, lr)
        loss_function: _Loss = get_loss_function(self.loss_function_id)

        lr_decay: float = (final_lr / lr) ** (1 / epochs) if final_lr is not None else 1.0
        scheduler: StepLR = StepLR(optimizer, step_size=1, gamma=lr_decay)

        model_state_manager = ModelStateManager(early_stopping)

        training_logger = ClassificationTrainingLogger(epochs)

        data_loader: DataLoader = DataLoader(self.dataset.train, batch_size=batch_size, shuffle=True)

        time_per_epoch: float = 0.0
        for epoch in range(epochs):
            batch_train_losses: list[float] = []
            start_time: float = time.time()
            for x_batch, y_batch in data_loader:
                optimizer.zero_grad()

                if self.shape is None:
                    self.shape = x_batch.shape[1:]

                x_batch = x_batch.to(self.device, dtype=self.dtype)
                y_batch = y_batch.to(self.device, dtype=torch.long)

                output: Tensor = self.model(x_batch)

                loss: Tensor = loss_function(output, y_batch)
                loss.backward()

                optimizer.step()

                batch_train_losses.append(loss.item())

            # END OF EPOCH
            scheduler.step()

            val_loss, val_accuracy = self._estimate_loss(batch_size)

            model_state_manager(self.model.state_dict(), val_loss)

            if model_state_manager.stop():
                self.logger.warning("early stopping")
                break

            time_per_epoch = time.time() - start_time

            training_logger.log(
                train_loss=np.mean(batch_train_losses).astype(float),
                val_loss=val_loss,
                train_accuracy=0.0,
                val_accuracy=val_accuracy,
                lr=scheduler.get_last_lr()[0],
                epoch=epoch + 1,
                time_per_epoch=time_per_epoch,
            )

            self.logger.info(training_logger, flush=True)

        # END OF TRAINING
        train_info: ClassificationTrainInfo = training_logger.get_info()

        self.logger.info(f'training completed | best loss: {train_info.best_val_loss:.4f}')

        model_state_manager.load_best_model_state(self.model)

        return train_info

    @torch.no_grad()
    def evaluate(self) -> ClassificationEvalInfo:
        if self.dataset.test is None and self.dataset.validation is not None:
            self.logger.warning("no test split found, using validation split for evaluation")
            self.dataset.test = self.dataset.validation
        else:
            raise ValueError(f"cannot evaluate without a test split")

        self.model.eval()

        batch_size: int = 128
        dataloader: DataLoader = DataLoader(self.dataset.test, batch_size=batch_size, shuffle=False)

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

    @torch.no_grad()
    def _estimate_loss(self, batch_size: int) -> tuple[float, float]:
        if self.dataset.validation is None:
            return np.inf, -np.inf

        self.model.eval()

        loss_function: _Loss | None = get_loss_function(self.loss_function_id)
        if loss_function is None:
            raise ValueError(f"invalid loss function id: {self.loss_function_id}")

        data_loader: DataLoader = DataLoader(self.dataset.validation, batch_size=batch_size, shuffle=False)

        losses: list[float] = []
        predictions: list[int] = []
        targets: list[int] = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device, dtype=self.dtype)
            y_batch = y_batch.to(self.device, dtype=torch.long)

            output: Tensor = self.model(x_batch)

            loss: Tensor = loss_function(output, y_batch)

            losses.append(loss.item())
            predictions.append(output.argmax(dim=1).cpu().numpy())
            targets.append(y_batch.cpu().numpy())

        self.model.train()

        val_loss: float = np.mean(losses).astype(float)
        val_accuracy: float = np.mean(np.concatenate(predictions) == np.concatenate(targets)).astype(float)

        return val_loss, val_accuracy

    def save(self, *,
             path: str,
             to_onnx: bool = False,
             dtype: str = "float32",
             shape: tuple | None = None,
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
            if self.shape is None and shape is None:
                raise ValueError("could not infer shape, please provide shape argument")

            shape = self.shape if self.shape is not None else shape
            dummy_input = torch.randn(1, *shape).to(model_device, model_dtype)  # type: ignore
            model_path = os.path.join(path, f"{model_name}.onnx")

            onnx.export(
                self.model,
                dummy_input,
                model_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"},
                              "output": {0: "prediction"}},
            )

        self.logger.info(f"saved model {model_name} to {path}")
