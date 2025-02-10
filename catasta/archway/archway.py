import platform
import os

import numpy as np

import torch
from torch import Tensor
from torch.nn import Identity
from torch.distributions import Distribution

from ..dataclasses import PredictionInfo
from ..log import ansi


def _get_device(device: str) -> torch.device:
    match device:
        case "cpu":
            return torch.device("cpu")
        case "cuda":
            return torch.device("cuda")
        case "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        case _:
            raise ValueError("Invalid device")


def _get_dtype(dtype: str) -> torch.dtype:
    match dtype:
        case "float16":
            return torch.float16
        case "float32":
            return torch.float32
        case "float64":
            return torch.float64
        case _:
            raise ValueError("Invalid dtype")


class Archway:
    """A class for inference with a trained model.
    """

    def __init__(self,
                 path: str,
                 device: str = "auto",
                 dtype: str = "float32",
                 verbose: bool = True,
                 ) -> None:
        """Initialize the Archway object.

        Arguments
        ---------
        path : str
            The path to the directory containing the saved model. The directory should be named after the model.
        device : str, optional
            The device to perform inference on. Can be "cpu", "cuda", or "auto".
        dtype : str, optional
            The data type to use for inference. Can be "float16", "float32", or "float64".
        verbose : bool, optional
            Whether to log information about the inference process.

        Raises
        ------
        ValueError
            If the load path is a file path.
        """
        self.path: str = path if path.endswith("/") else f"{path}/"

        self.device: torch.device = _get_device(device)
        self.dtype: torch.dtype = _get_dtype(dtype)

        self._load_model()

        if not verbose:
            return

        n_params: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        device_name: str = torch.cuda.get_device_name() if self.device.type == "cuda" else platform.processor()
        print(
            f"{ansi.BOLD}{ansi.GREEN}-> inference info{ansi.RESET}\n"
            f"   |> model: {self.model.__class__.__name__} ({n_params} trainable parameters)\n"
            f"   |> device: {self.device} ({device_name})"
        )

    @torch.no_grad()
    def predict(self, input: np.ndarray | Tensor) -> PredictionInfo:
        """Perform inference with the model.

        Arguments
        ---------
        input : np.ndarray or Tensor
            The input data to make predictions on.

        Returns
        -------
        ~catasta.dataclasses.PredictionInfo
            The predicted values and standard deviations.
        """
        input_tensor: Tensor = torch.tensor(input) if isinstance(input, np.ndarray) else input
        input_tensor = input_tensor.to(self.device, self.dtype)

        output = self.likelihood(self.model(input_tensor))

        if isinstance(output, Distribution):
            predicted_output = output.mean.cpu().numpy()
            predicted_std = output.stddev.cpu().numpy()
        else:
            predicted_output = output.cpu().numpy()
            predicted_std = np.zeros_like(predicted_output)

        if predicted_output.ndim == 2:
            argmax: np.ndarray = np.argmax(predicted_output, axis=1)
        else:
            argmax: np.ndarray = np.array([])

        return PredictionInfo(
            value=predicted_output,
            std=predicted_std,
            argmax=argmax,
        )

    def _load_model(self) -> None:
        if "." in self.path:
            raise ValueError("load path must be a directory, not a file path")

        self.model = Identity()
        self.likelihood = Identity()
        for model in os.listdir(self.path):
            if os.path.basename(self.path.rstrip("/")).lower() == model.split(".")[0].lower():
                self.model = torch.load(os.path.join(self.path, model), map_location=self.device)
            elif "likelihood" in model.lower():
                self.likelihood = torch.load(os.path.join(self.path, model), map_location=self.device)

        if isinstance(self.model, Identity):
            raise ValueError("model not found")

        self.model.to(self.device, self.dtype)
        self.likelihood.to(self.device, self.dtype)

        self.model.eval()
        self.likelihood.eval()
