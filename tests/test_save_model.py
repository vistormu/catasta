import os

from catasta.models import TransformerRegressor, ApproximateGPRegressor
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.dataclasses import RegressionEvalInfo
from catasta.transformations import (
    Normalization,
    Decimation,
    WindowSliding,
    Slicing,
    Custom,
)

from vclog import Logger


def vanilla() -> None:
    n_dim: int = 768
    dataset_root: str = "tests/data/nylon_elastic/strain/"
    input_trasnsformations = [
        Custom(lambda x: x[500_000:1_000_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=100),
        WindowSliding(window_size=n_dim, stride=1),
    ]
    output_trasnsformations = [
        Custom(lambda x: x[500_000:1_000_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=100),
        Slicing(amount=n_dim - 1, end="left"),
    ]

    n_files = len(os.listdir(dataset_root))
    train_split = (n_files-1) / n_files
    dataset = RegressionDataset(
        root=dataset_root,
        input_transformations=input_trasnsformations,
        output_transformations=output_trasnsformations,
        splits=(train_split, 1-train_split, 0.0)
    )

    model = TransformerRegressor(
        context_length=n_dim,
        n_patches=8,
        d_model=8,
        n_heads=4,
        n_layers=2,
        feedforward_dim=4,
        head_dim=4,
        dropout=0.0,
    )
    scaffold = RegressionScaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="mse",
    )

    scaffold.train(
        epochs=100,
        batch_size=256,
        lr=1e-4,
    )
    info: RegressionEvalInfo = scaffold.evaluate()
    Logger.debug(info)

    scaffold.save(path="tests/models/", to_onnx=True)


def gp():
    n_dim: int = 768
    dataset_root: str = "tests/data/nylon_elastic/strain/"
    input_trasnsformations = [
        Custom(lambda x: x[500_000:1_000_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=100),
        WindowSliding(window_size=n_dim, stride=1),
    ]
    output_trasnsformations = [
        Custom(lambda x: x[500_000:1_000_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=100),
        Slicing(amount=n_dim - 1, end="left"),
    ]

    n_files = len(os.listdir(dataset_root))
    train_split = (n_files-1) / n_files
    dataset = RegressionDataset(
        root=dataset_root,
        input_transformations=input_trasnsformations,
        output_transformations=output_trasnsformations,
        splits=(train_split, 0.0, 1-train_split)
    )

    model = ApproximateGPRegressor(
        context_length=n_dim,
        n_inducing_points=128,
    )
    scaffold = RegressionScaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="variational_elbo",
    )

    scaffold.train(
        epochs=100,
        batch_size=256,
        lr=1e-4,
    )
    info: RegressionEvalInfo = scaffold.evaluate()
    Logger.debug(info)

    scaffold.save(path="tests/models/gp/")


if __name__ == '__main__':
    # vanilla()
    gp()
