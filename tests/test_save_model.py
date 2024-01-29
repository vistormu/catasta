import os

import matplotlib.pyplot as plt

from catasta.models import ApproximateGPRegressor, TransformerRegressor
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.transformations import (
    Normalization,
    Decimation,
    WindowSliding,
    Slicing,
    Custom,
)


def main() -> None:
    n_dim: int = 16
    dataset_root: str = "tests/data/nylon_elastic/paper/strain/mixed_10_20/"
    input_trasnsformations = [
        Custom(lambda x: x[500_000:1_500_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=100),
        WindowSliding(window_size=n_dim, stride=1),
    ]
    output_trasnsformations = [
        Custom(lambda x: x[500_000:1_500_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=100),
        Slicing(amount=n_dim-1, end="left"),
    ]
    n_files: int = len(os.listdir(dataset_root))
    train_split: float = (n_files - 1) / n_files
    dataset = RegressionDataset(
        root=dataset_root,
        input_transformations=input_trasnsformations,
        output_transformations=output_trasnsformations,
        splits=(train_split, 1 - train_split, 0.0),
        # splits=(train_split, 0.0, 1 - train_split),
    )
    model = TransformerRegressor(
        context_length=n_dim,
        n_patches=2,
        d_model=64,
        n_heads=2,
        n_layers=4,
        feedforward_dim=32,
        head_dim=2,
        dropout=0.0,
        use_fft=True,
    )

    # model = ApproximateGPRegressor(
    #     n_inducing_points=16,
    #     n_inputs=n_dim,
    #     kernel="rq",
    #     mean="constant"
    # )
    scaffold = RegressionScaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="mse",
        # loss_function="variational_elbo",
        save_path="tests/models/",
    )

    scaffold.train(
        epochs=100,
        batch_size=256,
        lr=1e-3,
    )

    metrics = scaffold.evaluate()
    print(metrics)

    plt.plot(metrics.predicted_output, label="predicted")
    plt.show()


if __name__ == '__main__':
    main()
