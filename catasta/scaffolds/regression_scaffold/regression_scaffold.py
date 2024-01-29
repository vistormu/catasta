from torch.nn import Module
from gpytorch.models.gp import GP

from .regression_scaffold_interface import RegressionScaffold as IRegressionScaffold
from ...datasets import RegressionDataset
from .vanilla_regression_scaffold import VanillaRegressionScaffold
from .gaussian_regression_scaffold import GaussianRegressionScaffold


def RegressionScaffold(*,
                       model: Module,
                       dataset: RegressionDataset,
                       optimizer: str,
                       loss_function: str,
                       save_path: str | None = None,
                       ) -> IRegressionScaffold:
    '''
    Create a scaffold for regression tasks.

    Arguments
    ---------
    model: Module
        The model to train. It must be a Catasta regressor.

    dataset: RegressionDataset
        The dataset to train on.

    optimizer: str
        The ID of the optimizer to use. Available optimizers are:
            - "adam"
            - "adamw"
            - "sgd"
            - "lbfgs"
            - "rmsprop"
            - "rprop"
            - "adadelta"
            - "adagrad"
            - "adamax"
            - "asgd"
            - "sparse_adam"

    loss_function: str
        The ID of the loss function to use. Available loss functions are:
            - "mse"
            - "l1"
            - "smooth_l1"
            - "huber"
            - "poisson" 
            - "kl_div"
            - "variational_elbo" (only for Gaussian processes)
            - "predictive_log" (only for Gaussian processes)
            - "variational_marginal_log" (only for Gaussian processes)

    save_path: str | None
        The path where to save the model. If None, the model is not saved. It must be a directory.

    Returns
    -------
    RegressionScaffold
        The scaffold class to train the model.
    '''
    match model:
        case GP():
            return GaussianRegressionScaffold(
                model=model,
                dataset=dataset,
                optimizer=optimizer,
                loss_function=loss_function,
                save_path=save_path,
            )
        case _:
            return VanillaRegressionScaffold(
                model=model,
                dataset=dataset,
                optimizer=optimizer,
                loss_function=loss_function,
                save_path=save_path,
            )
