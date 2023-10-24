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

from ..datasets import ModelDataset
from ..entities import TrainInfo, EvalInfo

# tmp
import matplotlib.pyplot as plt


class GaussianRegressorScaffold:
    def __init__(self, model: Module, dataset: ModelDataset) -> None:
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype: torch.dtype = torch.float32

        self.model: Module = model.to(self.device)
        self.likelihood = GaussianLikelihood().to(self.device)
        self.dataset: ModelDataset = dataset

        self.logger: Logger = Logger("catasta")
        self.logger.info(f"using device: {self.device}")

        self.train_split: float = 0.8

    def train(self,
              epochs: int = 100,
              batch_size: int = 32,
              train_split: float = 0.8,
              lr: float = 1e-3,
              stop_condition: float | None = None,
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

                output: Tensor = self.model(x_batch)

                loss: Tensor = -mll(output, y_batch)  # type: ignore
                batch_losses.append(loss.item())
                loss.backward()

                optimizer.step()

                self.logger.info(f"epoch {i}/{epochs} | {int((i/epochs)*100+(j/len(data_loader))*100/epochs)}% | loss: {loss.item():.4f}", flush=True)

            losses.append(np.mean(batch_losses))

            # stop if the loss has not changed by more than stop_condition within the last 10 epochs with cumulative subtraction
            if stop_condition is not None and len(losses) > 10 and np.sum(np.diff(losses[-10:]) < stop_condition) == 0:
                self.logger.info(f"stopping early at epoch {i}")
                break

        self.logger.info(f'epoch {epochs}/{epochs} | 100% | loss: {np.min(losses):.4f}')

        return np.array(losses)

    @ torch.no_grad()
    def predict(self, input: np.ndarray | Tensor) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        self.likelihood.eval()

        if isinstance(input, np.ndarray):
            input_tensor: Tensor = torch.tensor(input, dtype=self.dtype, device=self.device)
        else:
            input_tensor: Tensor = input.to(self.device, dtype=self.dtype)

        output: MultivariateNormal = self.model(input_tensor)

        mean: np.ndarray = output.mean.cpu().numpy()
        std: np.ndarray = output.stddev.cpu().numpy()

        return mean, std

    @ torch.no_grad()
    def evaluate(self, plot_results: bool = False) -> EvalInfo:
        self.model.eval()
        self.likelihood.eval()

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

        if plot_results:
            plt.figure(figsize=(10, 10))
            plt.xlim(-0.2, 1.2)
            plt.ylim(-0.2, 1.2)
            plt.plot(test_x[:, -1].flatten(), test_y, 'k.')
            plt.plot(test_x[:, -1].flatten(), means, 'b.')
            plt.fill_between(np.mean(test_x, axis=1), means + 2*stds, means - 2*stds, color='b', alpha=0.2)
            plt.show()

            plt.figure(figsize=(30, 20))
            plt.plot(test_y, 'k')
            plt.plot(means, 'b')
            plt.fill_between(np.arange(test_x.shape[0]), means + 2*stds, means - 2*stds, color='b', alpha=0.2)
            plt.show()

        return EvalInfo(means, test_y)
