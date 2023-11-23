import numpy as np

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from torch.distributions import Distribution

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal

from vclog import Logger

from .regression_scaffold_interface import IRegressionScaffold
from ...utils import get_optimizer, get_objective_function
from ...datasets import RegressionDataset
from ...dataclasses import RegressionEvalInfo, RegressionTrainInfo, RegressionPrediction


class GaussianRegressionScaffold(IRegressionScaffold):
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
        self.logger.info(f"using device: {self.device}")

    def train(self,
              epochs: int = 100,
              batch_size: int = 32,
              lr: float = 1e-3,
              ) -> RegressionTrainInfo:
        self.model.train()
        self.likelihood.train()

        data_loader: DataLoader = DataLoader(self.dataset.train, batch_size=batch_size, shuffle=False)

        optimizer: Optimizer | None = get_optimizer(self.optimizer_id, [self.model, self.likelihood], lr)
        if optimizer is None:
            raise ValueError(f"invalid optimizer id: {self.optimizer_id}")

        mll: MarginalLogLikelihood | None = get_objective_function(self.loss_function_id, self.model, self.likelihood, len(self.dataset.train))
        if mll is None:
            raise ValueError(f"invalid loss function id: {self.loss_function_id}")

        losses: list[float] = []
        for i in range(epochs):
            batch_losses: list[float] = []
            for j, (x_batch, y_batch) in enumerate(data_loader):
                optimizer.zero_grad()

                x_batch = x_batch.to(self.device, dtype=self.dtype)
                y_batch = y_batch.to(self.device, dtype=self.dtype)

                output: MultivariateNormal = self.model(x_batch)

                loss: Tensor = -mll(output, y_batch)  # type: ignore
                batch_losses.append(loss.item())
                loss.backward()

                optimizer.step()

                self.logger.info(f"epoch {i}/{epochs} | {int((i/epochs)*100+(j/len(data_loader))*100/epochs)}% | loss: {loss.item():.4f}", flush=True)

            losses.append(np.mean(batch_losses).astype(float))

        self.logger.info(f'epoch {epochs}/{epochs} | 100% | loss: {np.min(losses):.4f}')

        return RegressionTrainInfo(np.array(losses))

    @torch.no_grad()
    def predict(self, input: np.ndarray | Tensor) -> RegressionPrediction:
        self.model.eval()
        self.likelihood.eval()

        input_tensor: Tensor = torch.tensor(input) if isinstance(input, np.ndarray) else input
        input_tensor = input_tensor.to(self.device, dtype=self.dtype)

        output: Distribution = self.likelihood(self.model(input_tensor))

        mean: np.ndarray = output.mean.cpu().numpy()
        std: np.ndarray = output.stddev.cpu().numpy()

        return RegressionPrediction(mean, std)

    @torch.no_grad()
    def evaluate(self) -> RegressionEvalInfo:
        test_index: int = np.floor(len(self.dataset) * self.dataset.splits[0]).astype(int)
        test_x: np.ndarray = self.dataset.inputs[test_index:]
        test_y: np.ndarray = self.dataset.outputs[test_index:].flatten()

        if self.dataset.test is None:
            raise ValueError(f"test split must be greater than 0")

        data_loader = DataLoader(self.dataset.test, batch_size=1, shuffle=False)

        means: np.ndarray = np.array([])
        stds: np.ndarray = np.array([])
        for x_batch, _ in data_loader:
            output: RegressionPrediction = self.predict(x_batch)
            means = np.concatenate([means, output.prediction])
            if output.stds is not None:
                stds = np.concatenate([stds, output.stds])

        if len(means) != len(test_y):
            min_len: int = min(len(means), len(test_y))
            means = means[-min_len:]
            stds = stds[-min_len:]
            test_y = test_y[-min_len:]

        return RegressionEvalInfo(test_x, test_y, means, stds)
