import os

from catasta.models import MambaFFTRegressor
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
    n_dim: int = 768
    dataset_root: str = "tests/data/nylon_elastic/strain/"
    input_trasnsformations = [
        Custom(lambda x: x[500_000:600_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=10),
        WindowSliding(window_size=n_dim, stride=1),
    ]
    output_trasnsformations = [
        Custom(lambda x: x[500_000:600_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=10),
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

    model = MambaFFTRegressor(
        context_length=n_dim,
        n_patches=2,
        n_layers=2,
        d_model=8,
        d_conv=4,
        d_state=4,
        expand=1,
    )
    scaffold = RegressionScaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="mse",
    )

    train_info: RegressionTrainInfo = scaffold.train(
        epochs=10,
        batch_size=256,
        lr=1e-3,
        final_lr=1e-4,
        early_stopping=(10, 0.01),
    )
    info: RegressionEvalInfo = scaffold.evaluate()

    Logger.debug(info)


if __name__ == '__main__':
    main()
