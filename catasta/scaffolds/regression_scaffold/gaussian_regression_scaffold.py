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
from ...dataclasses import RegressionEvalInfo, RegressionTrainInfo


class GaussianRegressionScaffold(RegressionScaffold):
    def __init__(self, *,
                 model: Module,
                 dataset: RegressionDataset,
                 optimizer: str = "adam",
                 loss_function: str = "variational_elbo",
                 save_path: str | None = None,
                 ) -> None:
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype: torch.dtype = torch.float32

        self.model: Module = model.to(self.device)
        self.likelihood = GaussianLikelihood().to(self.device)
        self.dataset: RegressionDataset = dataset

        self.optimizer_id: str = optimizer
        self.loss_function_id: str = loss_function

        if save_path is not None and "." in save_path:
            raise ValueError("save path must be a directory")
        self.save_path: str | None = save_path

        self.logger: Logger = Logger("catasta")

        # Logging info
        message: str = f"using {self.device} with {torch.cuda.get_device_name()}" if torch.cuda.is_available() else f"using {self.device}"
        self.logger.info(message)
        n_parameters: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"training model {self.model.__class__.__name__} ({n_parameters} parameters)")

    def train(self,
              epochs: int = 100,
              batch_size: int = 32,
              lr: float = 1e-3,
              final_lr: float | None = None,
              early_stopping: tuple[int, float] | None = None,
              verbose: bool = True,
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

        model_state_manager = ModelStateManager(early_stopping, self.save_path)

        data_loader: DataLoader = DataLoader(self.dataset.train, batch_size=batch_size, shuffle=False)

        time_per_epoch: float = 0.0

        losses: list[float] = []
        for i in range(epochs):
            batch_losses: list[float] = []
            start_time: float = time.time()
            for j, (x_batch, y_batch) in enumerate(data_loader):
                optimizer.zero_grad()

                x_batch = x_batch.to(self.device, dtype=self.dtype)
                y_batch = y_batch.to(self.device, dtype=self.dtype)

                output: MultivariateNormal = self.model(x_batch)

                loss: Tensor = -mll(output, y_batch)  # type: ignore
                loss.backward()

                optimizer.step()

                batch_losses.append(loss.item())

                if verbose:
                    log_train_data(
                        train_loss=losses[-1] if len(losses) > 0 else 0.0,
                        val_loss=0.0,
                        best_val_loss=0.0,
                        lr=scheduler.get_last_lr()[0],
                        epoch=i,
                        epochs=epochs,
                        percentage=int((i/epochs)*100+(j/len(data_loader))*100/epochs),
                        time_per_epoch=time_per_epoch,
                    )

            # END OF EPOCH
            scheduler.step()

            losses.append(np.mean(batch_losses).astype(float))

            model_state_manager(self.model.state_dict(), losses[-1])

            if model_state_manager.stop():
                self.logger.warning("early stopping")
                break

            time_per_epoch = time.time() - start_time

        # END OF TRAINING
        self.logger.info(f"training completed | best loss: {np.min(losses)}")

        model_state_manager.load_best_model_state(self.model)
        model_state_manager.save_models([self.model, self.likelihood])

        return RegressionTrainInfo(np.array(losses))

    @torch.no_grad()
    def evaluate(self) -> RegressionEvalInfo:
        if self.dataset.test is None:
            raise ValueError("test split is empty")

        self.model.eval()
        self.likelihood.eval()

        x, y = next(iter(DataLoader(self.dataset.test, batch_size=len(self.dataset.test), shuffle=False)))

        x: Tensor = x.to(self.device, dtype=self.dtype)
        y: Tensor = y.to(self.device, dtype=self.dtype)

        output: Distribution = self.likelihood(self.model(x))

        true_input: np.ndarray = x.cpu().numpy()[:, -1]
        true_output: np.ndarray = y.cpu().numpy()
        predicted_output: np.ndarray = output.mean.cpu().numpy()
        predicted_std: np.ndarray = output.stddev.cpu().numpy()

        return RegressionEvalInfo(true_input, true_output, predicted_output, predicted_std)
