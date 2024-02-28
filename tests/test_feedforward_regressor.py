import os
import numpy as np

from catasta.models import FeedforwardRegressor
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
    n_dim: int = 32
    dataset_root: str = "tests/data/nylon_elastic/strain/"
    input_trasnsformations = [
        Custom(lambda x: x[500_000:1_000_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=100),
        WindowSliding(window_size=n_dim, stride=1),
    ]
    output_trasnsformations = [
        Custom(lambda x: x[500_000:1_000_000]),
        Custom(lambda x: x-1),
        Normalization("minmax"),
        Decimation(decimation_factor=100),
        Slicing(amount=n_dim - 1, end="left"),
    ]
    n_files: int = len(os.listdir(dataset_root))
    train_split: float = (n_files - 1) / n_files
    splits = (train_split, 1 - train_split, 0.0)
    dataset = RegressionDataset(
        root=dataset_root,
        input_transformations=input_trasnsformations,
        output_transformations=output_trasnsformations,
        splits=splits,
    )

    model = FeedforwardRegressor(
        input_dim=n_dim,
        hidden_dims=[8, 16, 8],
        dropout=0.0,
        use_layer_norm=True,
        activation="relu",
    )
    scaffold = RegressionScaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="mse",
    )

    train_info: RegressionTrainInfo = scaffold.train(
        epochs=100,
        batch_size=256,
        lr=1e-3,
        final_lr=1e-4,
    )
    Logger.debug(f"min train loss: {train_info.best_train_loss:.4f}")

    info: RegressionEvalInfo = scaffold.evaluate()
    Logger.debug(info)


if __name__ == '__main__':
    main()
