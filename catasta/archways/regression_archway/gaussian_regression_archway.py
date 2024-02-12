import os

import numpy as np

import torch
from torch import Tensor
from torch.nn import Module
from torch.distributions import Distribution

from gpytorch.likelihoods import GaussianLikelihood

from vclog import Logger

from .regression_archway_interface import RegressionArchway
from ...dataclasses import RegressionPrediction


class GaussianRegressionArchway(RegressionArchway):
    def __init__(self, *,
                 model: Module,
                 path: str | None = None,
                 device: str = "auto",
                 ) -> None:
        if device == "auto":
            self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device == "cpu":
            self.device: torch.device = torch.device('cpu')
        elif device == "cuda":
            self.device: torch.device = torch.device('cuda')
        else:
            raise ValueError(f"Unknown device: {device}")

        self.dtype: torch.dtype = torch.float32

        self.logger = Logger("catasta")

        self.model: Module = model.to(self.device, dtype=self.dtype)
        self.likelihood = GaussianLikelihood().to(self.device)
        self.load_models(path)

        self.model.eval()
        self.likelihood.eval()

        # Logging info
        message: str = f"using {self.device} with {torch.cuda.get_device_name()}" if torch.cuda.is_available() else f"using {self.device}"
        self.logger.info(message)
        n_parameters: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"infering with {self.model.__class__.__name__} ({n_parameters} parameters)")

    def load_models(self, path: str | None) -> None:
        if path is None:
            return

        # get the .pt files in the directory
        files: list[str] = os.listdir(path)

        for file in files:
            if not file.endswith(".pt"):
                continue

            if self.model.__class__.__name__ in file:
                self.logger.info(f"Loading model from {file}")
                self.model.load_state_dict(torch.load(os.path.join(path, file)))
            elif self.likelihood.__class__.__name__ in file:
                self.logger.info(f"Loading likelihood from {file}")
                self.likelihood.load_state_dict(torch.load(os.path.join(path, file)))

    @torch.no_grad()
    def predict(self, input: np.ndarray | Tensor) -> RegressionPrediction:
        input_tensor: Tensor = torch.tensor(input) if isinstance(input, np.ndarray) else input
        input_tensor = input_tensor.to(self.device, dtype=self.dtype)

        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)

        output: Distribution = self.likelihood(self.model(input_tensor))

        mean: np.ndarray = output.mean.cpu().numpy()
        std: np.ndarray = output.stddev.cpu().numpy()

        return RegressionPrediction(mean, std)

    def __call__(self, input: np.ndarray | Tensor) -> RegressionPrediction:
        return self.predict(input)
