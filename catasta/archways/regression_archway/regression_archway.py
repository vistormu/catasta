from torch.nn import Module
from gpytorch.models.gp import GP

from .regression_archway_interface import RegressionArchway as IRegressionArchway
from .vanilla_regression_archway import VanillaRegressionArchway
from .gaussian_regression_archway import GaussianRegressionArchway


def RegressionArchway(*,
                      model: Module,
                      path: str | None = None,
                      device: str = "auto",
                      ) -> IRegressionArchway:
    '''
    Create an "archway" for regression inference.

    Arguments
    ---------
    model: Module
        The model to use for inference.

    path: str | None
        The path to the pretrained model. If None, the model is not loaded.

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
            )
