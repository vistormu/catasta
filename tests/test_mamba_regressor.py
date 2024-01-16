import numpy as np

from catasta.models import MambaRegressor
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.dataclasses import RegressionEvalInfo, RegressionTrainInfo
from catasta.transformations import Normalization, Decimation, WindowSliding, FIRFiltering, Slicing
from catasta.utils import Plotter

from vclog import Logger


def main() -> None:
    n_dim: int = 256
    dataset_root: str = "tests/data/nylon_carmen/paper/strain/mixed_10_20/"
    input_trasnsformations = [
        Slicing(amount=1, end="right"),
        Normalization("zscore"),
        Decimation(decimation_factor=100),
        FIRFiltering(fs=100, cutoff=0.1, numtaps=101),
        WindowSliding(window_size=n_dim, stride=1),
    ]
    output_trasnsformations = [
        Slicing(amount=1, end="right"),
        Normalization("zscore"),
        Decimation(decimation_factor=100),
        FIRFiltering(fs=100, cutoff=0.1, numtaps=101),
        Slicing(amount=n_dim - 1, end="left"),
    ]

    # n_dim: int = 256
    # dataset_root: str = "tests/data/wire_lisbeth/strain/"
    # input_trasnsformations = [
    #     Normalization("zscore"),
    #     WindowSliding(window_size=n_dim, stride=1),
    # ]
    # output_trasnsformations = [
    #     Normalization("zscore"),
    #     Slicing(amount=n_dim - 1, end="left"),
    # ]

    dataset = RegressionDataset(
        root=dataset_root,
        input_transformations=input_trasnsformations,
        output_transformations=output_trasnsformations,
        splits=(6/7, 1/7, 0.0),
    )
    model = MambaRegressor(
        context_length=n_dim,
        n_patches=32,
        d_model=16,
        d_state=8,
        d_conv=3,
        expand=10,
    )
    scaffold = RegressionScaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="mse",
    )
    train_info: RegressionTrainInfo = scaffold.train(
        epochs=500,
        batch_size=128,
        lr=1e-3,
        final_lr=1e-4,
        early_stopping=(10, 1e-3),
    )

    Logger.debug(f"min train loss: {np.min(train_info.train_loss):.4f}, "
                 f"min eval loss: {np.min(train_info.eval_loss):.4f}")  # type: ignore

    info: RegressionEvalInfo = scaffold.evaluate()

    Logger.debug(info)

    Plotter(
        train_info=train_info,
        eval_info=info,
    ).plot_all()


if __name__ == '__main__':
    main()
