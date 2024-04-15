import os

from catasta.models import TransformerRegressor
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


def main() -> None:
    n_dim: int = 768
    dataset_root: str = "data/nylon_wire/"
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

    train_info: RegressionTrainInfo = scaffold.train(
        epochs=10,
        batch_size=256,
        lr=1e-3,
    )
    print(f"min eval loss: {train_info.best_val_loss:.4f}")

    info: RegressionEvalInfo = scaffold.evaluate()

    print(info)


if __name__ == '__main__':
    main()
