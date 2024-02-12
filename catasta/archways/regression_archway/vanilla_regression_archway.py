import os

import torch
from torch import Tensor
from torch.nn import Module

import numpy as np

from vclog import Logger

from ...dataclasses import RegressionPrediction
from .regression_archway_interface import RegressionArchway


class VanillaRegressionArchway(RegressionArchway):
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
        self.load_model(path)

        self.model.eval()

        # Logging info
        message: str = f"using {self.device} with {torch.cuda.get_device_name()}" if torch.cuda.is_available() else f"using {self.device}"
        self.logger.info(message)
        n_parameters: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"infering with {self.model.__class__.__name__} ({n_parameters} parameters)")

    def load_model(self, path: str | None) -> None:
        if path is None:
            return

        files: list[str] = os.listdir(path)
        for file in files:
            if not file.endswith(".pt"):
                continue

            if self.model.__class__.__name__ in file:
                self.logger.info(f"Loading model from {file}")
                self.model.load_state_dict(torch.load(os.path.join(path, file)))

    @torch.no_grad()
    def predict(self, input: np.ndarray | Tensor) -> RegressionPrediction:
        input_tensor: Tensor = torch.tensor(input) if isinstance(input, np.ndarray) else input
        input_tensor = input_tensor.to(self.device, dtype=self.dtype)

        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)

        output: Tensor = self.model(input_tensor)

        return RegressionPrediction(output.cpu().numpy())

    def __call__(self, input: np.ndarray | Tensor) -> RegressionPrediction:
        return self.predict(input)
