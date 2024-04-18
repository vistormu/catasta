import platform
import os

import numpy as np

import torch
from torch import Tensor
from torch.nn import Module, Identity
from torch.distributions import Distribution

from gpytorch.models import GP

from vclog import Logger

from catasta.scaffolds.utils import get_likelihood
from ..dataclasses import PredictionInfo


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

    def __init__(self, *,
                 model: Module,
                 path: str | None = None,
                 likelihood: str | Module | None = None,
                 device: str = "auto",
                 dtype: str = "float32",
                 verbose: bool = True,
                 ) -> None:
        """Initialize the Archway object.

        Arguments
        ---------
        model : Module
            The model to perform inference with.
        path : str, optional
            The path to the directory containing the saved model. If None, the model is assumed to be loaded in memory.
        likelihood : str or Module, optional
            The likelihood function to use for the model. If None, the default likelihood for the model is used.
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
        self.path: str | None = path

        self.device: torch.device = _get_device(device)
        self.dtype: torch.dtype = _get_dtype(dtype)

        self.model: Module = model
        if isinstance(self.model, GP) and likelihood is None:
            self.likelihood: Module = get_likelihood("gaussian")
        else:
            self.likelihood: Module = get_likelihood(likelihood)

        self.logger: Logger = Logger("catasta", disable=not verbose)

        self._load_model()
        self._log_inference_info()

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
        if self.path is None:
            self.model.to(self.device, self.dtype)
            self.likelihood.to(self.device, self.dtype)
            return

        if "." in self.path:
            raise ValueError("load path must be a directory, not a file path")

        save_path: str = os.path.join(self.path, self.model.__class__.__name__)

        for model in [self.model, self.likelihood]:
            if isinstance(model, Identity):
                continue

            model_path: str = os.path.join(save_path, f"{model.__class__.__name__}.pt")
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.to(self.device, self.dtype)
            model.eval()

            self.logger.info(f"loaded {model.__class__.__name__} from {model_path}")

    def _log_inference_info(self) -> None:
        n_params: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        device_name: str = torch.cuda.get_device_name() if self.device.type == "cuda" else platform.processor()
        self.logger.info(f"""inferring
    -> model: {self.model.__class__.__name__} ({n_params} trainable parameters)
    -> device: {self.device} ({device_name})
        """)
