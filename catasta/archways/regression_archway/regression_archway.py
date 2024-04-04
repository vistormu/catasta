from torch.nn import Module
from gpytorch.models.gp import GP

from .regression_archway_interface import RegressionArchway as IRegressionArchway
from .vanilla_regression_archway import VanillaRegressionArchway
from .gaussian_regression_archway import GaussianRegressionArchway


def RegressionArchway(*,
                      model: Module | None = None,
                      path: str | None = None,
                      device: str = "auto",
                      dtype: str = "float32",
                      from_onnx: bool = False,
                      ) -> IRegressionArchway:
    '''
    Create an "archway" for regression inference.

    Arguments
    ---------
    model: Module
        The model to use for inference. If None, you must provide a path to a pretrained ONNX model.

    path: str | None
        The path to the pretrained model. If None, the model is not loaded.

    device: str
        The device to use for inference. Can be "auto", "cpu", or "cuda".

    dtype: str
        The data type to use for inference. Can be "float16", "float32", or "float64".

    from_onnx: bool
        Whether the model is loaded from an ONNX file.

    Returns
    -------
    RegressionArchway
        The archway class to perform inference.
    '''

    match model:
        case GP():
            return GaussianRegressionArchway(
                model=model,
                path=path,
                device=device,
            )
        case _:
            return VanillaRegressionArchway(
                model=model,
                path=path,
                device=device,
                dtype=dtype,
                from_onnx=from_onnx,
            )
