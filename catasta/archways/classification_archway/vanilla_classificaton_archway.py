import os

import torch
from torch import Tensor
from torch.nn import Module

import onnxruntime as ort

import numpy as np

from vclog import Logger

from ...dataclasses import ClassificationPrediction


class VanillaClassificationArchway:
    def __init__(self, *,
                 model: Module | None = None,
                 path: str | None = None,
                 from_onnx: bool = False,
                 device: str = "auto",
                 dtype: str = "float32",
                 ) -> None:
        # Check args
        if model is None and path is None:
            raise ValueError("model or path must be provided")

        if not device in ["auto", "cpu", "cuda"]:
            raise ValueError(f"Unknown device: {device}")

        if not dtype in ["float16", "float32", "float64"]:
            raise ValueError(f"Unknown dtype: {dtype}")

        if path is not None and "." in path:
            raise ValueError("path must be a directory")

        if path is not None and not os.path.exists(path):
            raise FileNotFoundError(f"directory not found: {path}")

        # Attributes
        self.from_onnx: bool = from_onnx

        if device == "auto":
            self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device == "cpu":
            self.device: torch.device = torch.device('cpu')
        elif device == "cuda":
            self.device: torch.device = torch.device('cuda')
        else:
            raise ValueError(f"Unknown device: {device}")

        self.dtype: torch.dtype = torch.float16 if dtype == "float16" else torch.float32 if dtype == "float32" else torch.float64
        self.np_dtype = np.float16 if dtype == "float16" else np.float32 if dtype == "float32" else np.float64

        # Load model
        self.model: Module | ort.InferenceSession = self._load_model(model, path)

        # Logging info
        self.logger = Logger("catasta")

        message: str = f"using {self.device} with {torch.cuda.get_device_name()}" if torch.cuda.is_available() else f"using {self.device}"
        self.logger.info(message)

    def _load_model(self, model: Module | None, path: str | None) -> Module | ort.InferenceSession:
        # prediction with provided model
        if model is not None and path is None:
            model.eval()
            return model.to(self.device, self.dtype)

        # prediction with model from onnx file
        if self.from_onnx and model is None and path is not None:
            for file in os.listdir(path):
                if file.endswith(".onnx"):
                    return ort.InferenceSession(os.path.join(path, file))

        # prediction with model from torch file
        if not self.from_onnx and model is not None and path is not None:
            model_path: str = os.path.join(path, f"{model.__class__.__name__}.pt")
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            return model.to(self.device, self.dtype)

        raise ValueError("could not load model. Check the provided arguments")

    @torch.no_grad()
    def predict(self, input: np.ndarray | Tensor) -> ClassificationPrediction:
        if self.from_onnx:
            output: np.ndarray = self._onnx_predict(input)
        else:
            output: np.ndarray = self._pytorch_predict(input)

        return ClassificationPrediction(output, np.argmax(output, axis=1))

    @torch.no_grad()
    def _pytorch_predict(self, input: np.ndarray | Tensor) -> np.ndarray:
        input_tensor: Tensor = torch.tensor(input) if isinstance(input, np.ndarray) else input
        input_tensor = input_tensor.to(self.device, self.dtype)

        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)

        output: Tensor = self.model(input_tensor)  # type: ignore

        return output.cpu().numpy()

    @torch.no_grad()
    def _onnx_predict(self, input: np.ndarray | Tensor) -> np.ndarray:
        model_input: np.ndarray = input if isinstance(input, np.ndarray) else input.cpu().numpy()
        if len(model_input.shape) == 1:
            model_input = model_input.reshape(1, -1)

        output: np.ndarray = self.model.run(["output"], {"input": model_input.astype(self.np_dtype)})[0]

        return output
