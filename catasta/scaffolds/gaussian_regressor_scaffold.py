import numpy as np

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.distributions import MultivariateNormal

from vclog import Logger

from ..datasets import RegressorDataset
from ..entities import EvalInfo


class GaussianRegressorScaffold:
    def __init__(self, model: Module, dataset: RegressorDataset) -> None:
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype: torch.dtype = torch.float32

        self.model: Module = model.to(self.device)
        self.likelihood = GaussianLikelihood().to(self.device)
        self.dataset: RegressorDataset = dataset

        self.logger: Logger = Logger("catasta")
        self.logger.info(f"using device: {self.device}")

        self.train_split: float = 0.8

    def train(self,
              epochs: int = 100,
              batch_size: int = 32,
              train_split: float = 0.8,
              lr: float = 1e-3,
              ) -> np.ndarray:
        self.model.train()
        self.likelihood.train()

        self.train_split = train_split

        train_dataset: Subset = Subset(self.dataset, range(int(len(self.dataset) * train_split)))
        data_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        optimizer = Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=lr)

        mll = VariationalELBO(self.likelihood, self.model, num_data=len(train_dataset))

        losses: list[float] = []
        for i in range(epochs):
            batch_losses: list[float] = []
            for j, (x_batch, y_batch) in enumerate(data_loader):
                optimizer.zero_grad()

                x_batch = x_batch.to(self.device, dtype=self.dtype)
                y_batch = y_batch.to(self.device, dtype=self.dtype)

                output: Tensor = self.model(x_batch)

                loss: Tensor = -mll(output, y_batch)  # type: ignore
                batch_losses.append(loss.item())
                loss.backward()

                optimizer.step()

                self.logger.info(f"epoch {i}/{epochs} | {int((i/epochs)*100+(j/len(data_loader))*100/epochs)}% | loss: {loss.item():.4f}", flush=True)

            losses.append(np.mean(batch_losses))

        self.logger.info(f'epoch {epochs}/{epochs} | 100% | loss: {np.min(losses):.4f}')

        return np.array(losses)

    @ torch.no_grad()
    def predict(self, input: np.ndarray | Tensor) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        self.likelihood.eval()

        input_tensor: Tensor = torch.tensor(input) if isinstance(input, np.ndarray) else input
        input_tensor = input_tensor.to(self.device, dtype=self.dtype)

        output: MultivariateNormal = self.model(input_tensor)

        mean: np.ndarray = output.mean.cpu().numpy()
        std: np.ndarray = output.stddev.cpu().numpy()

        return mean, std

    @ torch.no_grad()
    def evaluate(self) -> EvalInfo:
        test_x: np.ndarray = self.dataset.inputs[int(len(self.dataset) * self.train_split)+1:]
        test_y: np.ndarray = self.dataset.outputs[int(len(self.dataset) * self.train_split)+1:]

        test_dataset = Subset(self.dataset, range(int(len(self.dataset) * self.train_split)+1, len(self.dataset)))
        data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        means: np.ndarray = np.array([])
        stds: np.ndarray = np.array([])
        for x_batch, _ in data_loader:
            mean, std = self.predict(x_batch)
            means = np.concatenate([means, mean])
            stds = np.concatenate([stds, std])

        return EvalInfo(test_x, test_y, means, stds)
