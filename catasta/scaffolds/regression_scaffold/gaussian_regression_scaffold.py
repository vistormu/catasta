import time

import numpy as np

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR

from torch.distributions import Distribution

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal

from vclog import Logger

from .regression_scaffold_interface import RegressionScaffold
from ...utils import get_optimizer, get_objective_function, ModelStateManager, log_train_data
from ...datasets import RegressionDataset
from ...dataclasses import RegressionEvalInfo, RegressionTrainInfo, RegressionPrediction


class GaussianRegressionScaffold(RegressionScaffold):
    def __init__(self, *,
                 model: Module,
                 dataset: RegressionDataset,
                 optimizer: str = "adam",
                 loss_function: str = "variational_elbo",
                 ) -> None:
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype: torch.dtype = torch.float32

        self.model: Module = model.to(self.device)
        self.likelihood = GaussianLikelihood().to(self.device)
        self.dataset: RegressionDataset = dataset

        self.optimizer_id: str = optimizer
        self.loss_function_id: str = loss_function

        self.logger: Logger = Logger("catasta")

        # Logging info
        message: str = f"using {self.device} with {torch.cuda.get_device_name()}" if torch.cuda.is_available() else f"using {self.device}"
        self.logger.info(message)
        n_parameters: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"training model {self.model.__class__.__name__} with {n_parameters} parameters")

    def train(self,
              epochs: int = 100,
              batch_size: int = 32,
              lr: float = 1e-3,
              final_lr: float | None = None,
              early_stopping: tuple[int, float] | None = None,
              ) -> RegressionTrainInfo:
        self.model.train()
        self.likelihood.train()

        optimizer: Optimizer | None = get_optimizer(self.optimizer_id, [self.model, self.likelihood], lr)
        if optimizer is None:
            raise ValueError(f"invalid optimizer id: {self.optimizer_id}")

        mll: MarginalLogLikelihood | None = get_objective_function(self.loss_function_id, self.model, self.likelihood, len(self.dataset.train))
        if mll is None:
            raise ValueError(f"invalid loss function id: {self.loss_function_id}")

        lr_decay: float = (final_lr / lr) ** (1 / epochs) if final_lr is not None else 1.0
        scheduler: StepLR = StepLR(optimizer, step_size=1, gamma=lr_decay)

        model_state_manager = ModelStateManager(early_stopping)

        data_loader: DataLoader = DataLoader(self.dataset.train, batch_size=batch_size, shuffle=False)

        time_per_batch: float = 0.0
        time_per_epoch: float = 0.0

        losses: list[float] = []
        for i in range(epochs):
            times_per_batch: list[float] = []
            batch_losses: list[float] = []
            for j, (x_batch, y_batch) in enumerate(data_loader):
                start_time: float = time.time()

                optimizer.zero_grad()

                x_batch = x_batch.to(self.device, dtype=self.dtype)
                y_batch = y_batch.to(self.device, dtype=self.dtype)

                output: MultivariateNormal = self.model(x_batch)

                loss: Tensor = -mll(output, y_batch)  # type: ignore
                loss.backward()

                optimizer.step()

                batch_losses.append(loss.item())
                times_per_batch.append((time.time() - start_time) * 1000)

                log_train_data(
                    train_loss=losses[-1] if len(losses) > 0 else 0.0,
                    val_loss=0.0,
                    best_val_loss=0.0,
                    lr=scheduler.get_last_lr()[0],
                    epoch=i,
                    epochs=epochs,
                    percentage=int((i/epochs)*100+(j/len(data_loader))*100/epochs),
                    time_per_batch=time_per_batch,
                    time_per_epoch=time_per_epoch,
                )

            # END OF EPOCH
            scheduler.step()

            losses.append(np.mean(batch_losses).astype(float))
            time_per_batch = np.mean(times_per_batch).astype(float)
            time_per_epoch = np.sum(times_per_batch).astype(float)

            model_state_manager(self.model.state_dict(), losses[-1])

            if model_state_manager.stop():
                self.logger.warning("early stopping")
                break

        # END OF TRAINING
        self.logger.info(f'epoch {epochs}/{epochs} | 100% | loss: {np.min(losses):.4f}')

        # self.model.load_state_dict(model_state_manager.get_best_model_state())

        return RegressionTrainInfo(np.array(losses))

    @torch.no_grad()
    def predict(self, input: np.ndarray | Tensor) -> RegressionPrediction:
        self.model.eval()
        self.likelihood.eval()

        input_tensor: Tensor = torch.tensor(input) if isinstance(input, np.ndarray) else input
        input_tensor = input_tensor.to(self.device, dtype=self.dtype)

        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)

        output: Distribution = self.likelihood(self.model(input_tensor))

        mean: np.ndarray = output.mean.cpu().numpy()
        std: np.ndarray = output.stddev.cpu().numpy()

        return RegressionPrediction(mean, std)

    @torch.no_grad()
    def evaluate(self) -> RegressionEvalInfo:
        if self.dataset.test is None:
            raise ValueError("test split is empty")

        true_input: np.ndarray = np.array([])
        true_output: np.ndarray = np.array([])
        predicted_output: np.ndarray = np.array([])
        predicted_std: np.ndarray = np.array([])
        for x, y in self.dataset.test:
            true_input = np.append(true_input, x[-1])
            true_output = np.append(true_output, y)

            output: RegressionPrediction = self.predict(x)
            predicted_output = np.append(predicted_output, output.prediction)
            predicted_std = np.append(predicted_std, output.stds if output.stds is not None else np.zeros_like(output.prediction))

        return RegressionEvalInfo(true_input, true_output, predicted_output, predicted_std)
