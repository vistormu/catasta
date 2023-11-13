from torch.nn import Module

from .regression_scaffold_interface import IRegressionScaffold
from ...datasets import RegressionDataset
from ...models import ApproximateGPRegressor, FeedforwardRegressor, TransformerRegressor
from .vanilla_regression_scaffold import VanillaRegressionScaffold
from .gaussian_regression_scaffold import GaussianRegressionScaffold
from .transformer_regression_scaffold import TransformerRegressionScaffold


def RegressionScaffold(model: Module, dataset: RegressionDataset, optimizer: str, loss_function: str) -> IRegressionScaffold:
    match model:
        case FeedforwardRegressor():
            return VanillaRegressionScaffold(
                model=model,
                dataset=dataset,
                optimizer=optimizer,
                loss_function=loss_function,
            )
        case ApproximateGPRegressor():
            return GaussianRegressionScaffold(
                model=model,
                dataset=dataset,
                optimizer=optimizer,
                loss_function=loss_function,
            )
        case TransformerRegressor():
            return TransformerRegressionScaffold(
                model=model,
                dataset=dataset,
                optimizer=optimizer,
                loss_function=loss_function,
            )
        case _:
            raise TypeError(f"Unknown model type: {type(model)}")
