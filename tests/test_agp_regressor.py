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
    Custom,
)

from vclog import Logger


def main() -> None:
    n_inducing_points: int = 16
    n_dim: int = 16
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
    dataset = RegressionDataset(
        root=dataset_root,
        input_transformations=input_trasnsformations,
        output_transformations=output_trasnsformations,
        splits=(8/9, 0.0, 1/9),
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
        epochs=100,
        batch_size=256,
        lr=1e-3,
        final_lr=1e-4,
    )
    Logger.debug(f"min train loss: {np.min(train_info.train_loss):.4f}")

    info: RegressionEvalInfo = scaffold.evaluate()
    Logger.debug(info)


if __name__ == '__main__':
    main()
