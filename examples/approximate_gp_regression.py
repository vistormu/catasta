import numpy as np

from catasta.models import ApproximateGPRegressor
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.dataclasses import RegressionEvalInfo, RegressionTrainInfo
from catasta.transformations import (
    Normalization,
    Decimation,
    WindowSliding,
    Slicing,
)


def main() -> None:
    # Hyperparameters
    n_dim: int = 16
    n_inducing_points: int = 128
    kernel: str = "rq"
    mean: str = "constant"

    optimizer: str = "adamw"
    loss_function: str = "variational_elbo"

    epochs: int = 300
    batch_size: int = 128
    lr: float = 1e-3
    final_lr: float = 1e-4
    early_stopping: tuple[int, float] = (10, 1e-4)

    # Dataset
    dataset_root: str = "examples/data/"
    input_trasnsformations = [
        Slicing(amount=1, end="right"),
        Normalization("minmax"),
        Decimation(decimation_factor=100),
        WindowSliding(window_size=n_dim, stride=1),
    ]
    output_trasnsformations = [
        Slicing(amount=1, end="right"),
        Normalization("minmax"),
        Decimation(decimation_factor=100),
        Slicing(amount=n_dim - 1, end="left"),
    ]
    dataset = RegressionDataset(
        root=dataset_root,
        input_transformations=input_trasnsformations,
        output_transformations=output_trasnsformations,
        splits=(6/7, 0.0, 1/7),
    )

    # Model
    model = ApproximateGPRegressor(
        n_inducing_points=n_inducing_points,
        n_inputs=n_dim,
        kernel=kernel,
        mean=mean,
    )

    # Scaffold
    scaffold = RegressionScaffold(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        loss_function=loss_function,
    )

    train_info: RegressionTrainInfo = scaffold.train(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        final_lr=final_lr,
        early_stopping=early_stopping,
    )
    print(f"min train loss: {np.min(train_info.train_loss):.4f}")

    info: RegressionEvalInfo = scaffold.evaluate()
    print(info)


if __name__ == '__main__':
    main()
