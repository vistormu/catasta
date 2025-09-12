import os
from typing import Any

import numpy as np

import torch
from torch.nn import Module
from torch import Tensor

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
        case "qint8":
            return torch.qint8
        case "quint8":
            return torch.quint8
        case "qint32":
            return torch.qint32
        case "float16":
            return torch.float16
        case "float32":
            return torch.float32
        case "float64":
            return torch.float64
        case _:
            raise ValueError("Invalid dtype")


def _compile_model(model: Module, compile_method: str, **kwargs) -> Module:
    match compile_method:
        case "none":
            return model

        case "torchscript":
            return torch.jit.script(model)

        case "torchscript_optimized":
            return torch.jit.optimize_for_inference(torch.jit.script(model))

        case "torchscript_quantized":
            import platform
            backend = "qnnpack" if "arm" in platform.processor().lower() else "fbgemm"
            torch.backends.quantized.engine = backend

            return torch.quantization.quantize_dynamic(
                model,
                {*kwargs["layers"]},
                dtype=_get_dtype(kwargs["dtype"]),
            )

        case "torchscript_quantized_optimized":
            import platform
            backend = "qnnpack" if "arm" in platform.processor().lower() else "fbgemm"
            torch.backends.quantized.engine = backend

            return torch.jit.optimize_for_inference(
                torch.quantization.quantize_dynamic(
                    model,
                    {*kwargs["layers"]},
                    dtype=kwargs["dtype"],
                )
            )

        case _:
            raise ValueError("Invalid compile method")


def _load_model(path: str, device: torch.device, dtype: torch.dtype) -> Module:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")

    if path.endswith(".pt") or path.endswith(".pth"):
        model = torch.load(path, map_location=device, weights_only=False)
        model.to(device, dtype)
        model.eval()

        print(
            f"{ansi.BOLD}{ansi.GREEN}-> model loaded{ansi.RESET}\n"
            f"   |> path: {path}"
        )

        return model

    if path.endswith(".onnx"):
        try:
            import onnxruntime as ort  # type: ignore
            ort.set_default_logger_severity(3)  # TMP: suppress onnxruntime warnings
        except ImportError:
            print(
                f"{ansi.RED}-> missing library{ansi.RESET}\n"
                f"   |> this method is only available if you have onnxruntime installed\n"
                f"   |> run: pip install onnxruntime"
            )
            raise ImportError("onnxruntime not installed")

        model = ort.InferenceSession(path)

        print(
            f"{ansi.BOLD}{ansi.GREEN}-> model loaded{ansi.RESET}\n"
            f"   |> path: {path}"
        )

        return model

    raise ValueError("Invalid model format")


class Archway:
    """a class for inference with a trained model"""

    def __init__(self,
                 path: str,
                 compile_method: str = "none",
                 device: str = "auto",
                 dtype: str = "float32",
                 quantization_kwargs: dict = {},
                 ) -> None:
        """initialize the Archway class

        arguments
        ---------
        path : str
            the path to the model file
        compile_method : str
            the method to compile the model. options: "none", "torchscript", "torchscript_optimized", "torchscript_quantized", "torchscript_quantized_optimized"
        device : str
            the device to run the model on. options: "cpu", "cuda", "auto"
        dtype : str
            the datatype to use for the model. options: "qint8", "quint8", "qint32", "float16", "float32", "float64"
        quantization_kwargs : dict
            keyword arguments for quantization: "layers" (set of layers to quantize), "dtype" (datatype to quantize to)

        raises
        ------
        """
        self.device = _get_device(device)
        self.dtype = _get_dtype(dtype)

        self.model: Module = _load_model(path, self.device, self.dtype)
        self.model = _compile_model(self.model, compile_method, **quantization_kwargs)

        self.predict = self.predict_torch if isinstance(self.model, Module) else self.predict_onnx

    def predict_torch(self, input: np.ndarray | Tensor) -> Any:
        """predict with a PyTorch model

        arguments
        ---------
        input : np.ndarray | Tensor
            the input data

        returns
        -------
        np.ndarray
            the predicted values
        """
        if isinstance(input, np.ndarray):
            input = torch.tensor(input).to(self.device, self.dtype)

        with torch.inference_mode():
            return self.model(input)

    def predict_onnx(self, input: np.ndarray) -> np.ndarray:
        """predict with an ONNX model

        arguments
        ---------
        input : np.ndarray
            the input data

        returns
        -------
        np.ndarray
            the predicted values
        """
        return self.model.run(None, {"input": input})[0]  # type: ignore

    def serve(self,
              host: str,
              port: int,
              pydantic_model: type,
              endpoint: str = "/predict",
              ) -> None:
        """serve the model with FastAPI

        arguments
        ---------
        host : str
            the host to run the server on
        port : int
            the port to run the server on
        pydantic_model : type
            a Pydantic model for the input data
        endpoint : str
            the name of the endpoint to serve the model on

        raises
        ------
        ImportError
            if the required libraries are not installed: uvicorn, pydantic, fastapi for the server and onnxruntime for ONNX models
        TypeError
            if pydantic_model is not a subclass of pydantic
        """
        try:
            import uvicorn  # type: ignore
            from pydantic import BaseModel  # type: ignore
            from fastapi import FastAPI, HTTPException  # type: ignore
        except ImportError:
            print(
                f"{ansi.RED}-> missing libraries for server{ansi.RESET}\n"
                f"   |> this method is only available if you have uvicorn, pydantic, and fastapi installed\n"
                f"   |> run: pip install uvicorn pydantic fastapi"
            )
            raise ImportError("missing libraries for server")

        if not issubclass(pydantic_model, BaseModel):
            raise TypeError("pydantic_model must be a subclass of pydantic.BaseModel")

        app = FastAPI()

        @app.post(endpoint)
        async def predict(data: pydantic_model) -> dict:  # type: ignore
            try:
                input = np.array([list(data.dict().values())]).astype(np.float32)
                output = self.predict(input)
                return {"output": output.tolist()}

            except Exception as e:
                print(e)
                raise HTTPException(status_code=500, detail=str(e))

        print()
        uvicorn.run(app, host=host, port=port)
