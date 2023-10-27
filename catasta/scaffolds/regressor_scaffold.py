import numpy as np

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim import Adam
from torch.nn.functional import mse_loss

from vclog import Logger

from ..datasets import ModelDataset
from ..entities import EvalInfo


class RegressorScaffold:
    def __init__(self, model: Module, dataset: ModelDataset) -> None:
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype: torch.dtype = torch.float32

        self.model: Module = model.to(self.device)
        self.dataset: ModelDataset = dataset

        self.logger: Logger = Logger("catasta")
        self.logger.info(f"using device: {self.device}")

        self.train_split: float = 0.8

    def train(self, *,
              epochs: int = 100,
              batch_size: int = 128,
              train_split: float = 0.8,
              lr: float = 1e-3,
              ) -> np.ndarray:
        self.model.train()

        self.train_split = train_split

        train_dataset: Subset = Subset(self.dataset, range(int(len(self.dataset) * train_split)))
        val_dataset: Subset = Subset(self.dataset, range(int(len(self.dataset) * train_split), len(self.dataset)))
        data_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        optimizer: Adam = Adam(self.model.parameters(), lr=lr)

        best_loss: float = np.inf
        best_model_state_dict: dict = self.model.state_dict()

        eval_losses: list[float] = []
        for i in range(epochs):
            batch_losses: list[float] = []
            for j, (x_batch, y_batch) in enumerate(data_loader):
                optimizer.zero_grad()

                x_batch: Tensor = x_batch.to(self.device, dtype=self.dtype)
                y_batch: Tensor = y_batch.to(self.device, dtype=self.dtype)

                output: Tensor = self.model(x_batch)

                loss: Tensor = mse_loss(output, y_batch)
                loss.backward()

                optimizer.step()

                batch_losses.append(loss.item())

                self.logger.info(f"epoch {i}/{epochs} | {int((i/epochs)*100+(j/len(data_loader))*100/epochs)}% | train loss: {loss.item():.4f} | eval loss: {best_loss:.4f}", flush=True)

            eval_loss: float = self._estimate_loss(val_dataset, batch_size=batch_size)
            eval_losses.append(eval_loss)

            if eval_loss < best_loss:
                best_loss = eval_loss
                best_model_state_dict = self.model.state_dict()

        self.logger.info(f'epoch {epochs}/{epochs} | 100% | loss: {np.min(eval_losses):.4f}')

        self.model.load_state_dict(best_model_state_dict)

        return np.array(eval_losses)

    @ torch.no_grad()
    def predict(self, input: np.ndarray | Tensor) -> np.ndarray:
        self.model.eval()

        input_tensor: Tensor = torch.tensor(input) if isinstance(input, np.ndarray) else input
        input_tensor = input_tensor.to(self.device, dtype=self.dtype)

        output: Tensor = self.model(input_tensor)

        return output.cpu().numpy()

    @ torch.no_grad()
    def evaluate(self) -> EvalInfo:
        test_x: np.ndarray = self.dataset.inputs[int(len(self.dataset) * self.train_split)+1:]
        test_y: np.ndarray = self.dataset.outputs[int(len(self.dataset) * self.train_split)+1:]

        test_dataset = Subset(self.dataset, range(int(len(self.dataset) * self.train_split)+1, len(self.dataset)))
        data_loader: DataLoader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        predictions: np.ndarray = np.array([])
        for x_batch, _ in data_loader:
            output: np.ndarray = self.predict(x_batch)
            predictions = np.concatenate((predictions, output.flatten()))

        return EvalInfo(test_x, test_y, predictions)

    @ torch.no_grad()
    def _estimate_loss(self, val_dataset: Dataset, batch_size: int) -> float:
        self.model.eval()

        data_loader: DataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        losses: list[float] = []
        for x_batch, y_batch in data_loader:
            x_batch: Tensor = x_batch.to(self.device, dtype=self.dtype)
            y_batch: Tensor = y_batch.to(self.device, dtype=self.dtype)

            output: Tensor = self.model(x_batch)

            loss: Tensor = mse_loss(output, y_batch)

            losses.append(loss.item())

        self.model.train()

        return np.mean(losses)
