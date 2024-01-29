import numpy as np

from catasta.models import ApproximateGPRegressor
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.dataclasses import RegressionEvalInfo, RegressionTrainInfo
from catasta.transformations import (
    Normalization,
    Decimation,
    WindowSliding,
    FIRFiltering,
    Slicing,
    DiffAndConcat,
    Custom,
)
from catasta.utils import Plotter

from vclog import Logger


def main() -> None:
    n_inducing_points: int = 256
    n_dim: int = 96
    dataset_root: str = "tests/data/nylon_carmen/paper/strain/mixed_10_20/"
    input_trasnsformations = [
        Slicing(amount=1, end="right"),
        Custom(lambda x: x-1),
        Normalization("maxabs"),
        Decimation(decimation_factor=100),
        FIRFiltering(fs=100, cutoff=0.1, numtaps=101),
        WindowSliding(window_size=n_dim, stride=1),
    ]
    output_trasnsformations = [
        Slicing(amount=1, end="right"),
        Normalization("minmax"),
        Decimation(decimation_factor=100),
        FIRFiltering(fs=100, cutoff=0.1, numtaps=101),
        Slicing(amount=n_dim - 1, end="left"),
    ]

    # n_inducing_points: int = 256
    # dataset_root: str = "tests/data/nylon_carmen/paper/strain/mixed_10_20/"
    # elements_per_diff: int = 64
    # n_diffs: int = 2
    # n_dim: int = elements_per_diff * (n_diffs + 1)
    # input_trasnsformations = [
    #     Slicing(amount=1, end="right"),
    #     # Normalization("zscore"),
    #     Decimation(decimation_factor=100),
    #     FIRFiltering(fs=100, cutoff=0.1, numtaps=101),
    #     DiffAndConcat(n_diffs=n_diffs, elements_per_diff=elements_per_diff, filter=FIRFiltering(fs=100, cutoff=0.1, numtaps=101)),
    #     WindowSliding(window_size=n_dim, stride=n_dim),
    # ]
    # output_trasnsformations = [
    #     Slicing(amount=1, end="right"),
    #     # Normalization("zscore"),
    #     Decimation(decimation_factor=100),
    #     FIRFiltering(fs=100, cutoff=0.1, numtaps=101),
    #     Slicing(amount=elements_per_diff, end="left"),
    # ]

    # n_inducing_points: int = 16
    # n_dim: int = 128
    # dataset_root: str = "tests/data/wire_lisbeth/strain/"
    # input_trasnsformations = []
    # output_trasnsformations = []

    dataset = RegressionDataset(
        root=dataset_root,
        input_transformations=input_trasnsformations,
        output_transformations=output_trasnsformations,
        splits=(6/7, 0.0, 1/7),
    )
    model = ApproximateGPRegressor(
        n_inducing_points=n_inducing_points,
        n_inputs=n_dim,
        kernel="rq",
        mean="constant"
    )
    scaffold = RegressionScaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="variational_elbo",
    )

    train_info: RegressionTrainInfo = scaffold.train(
        epochs=300,
        batch_size=128,
        lr=1e-3,
        final_lr=1e-4,
        early_stopping=(10, 1e-4),
    )
    Logger.debug(f"min train loss: {np.min(train_info.train_loss):.4f}")

    info: RegressionEvalInfo = scaffold.evaluate()
    Logger.debug(info)

    plotter = Plotter(
        train_info=train_info,
        eval_info=info,
    )
    plotter.plot_all()


if __name__ == '__main__':
    main()
