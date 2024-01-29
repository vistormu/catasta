import numpy as np

from catasta.models import TransformerRegressor
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.dataclasses import RegressionEvalInfo, RegressionTrainInfo
from catasta.transformations import Normalization, Decimation, WindowSliding, FIRFiltering, Slicing

from vclog import Logger


def main() -> None:
    n_dim: int = 32
    dataset_root: str = "tests/data/nylon_carmen/paper/strain/mixed_10_20/"
    input_trasnsformations = [
        Slicing(amount=1, end="right"),
        Normalization("minmax"),
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
    dataset = RegressionDataset(
        root=dataset_root,
        input_transformations=input_trasnsformations,
        output_transformations=output_trasnsformations,
        splits=(6/7, 1/7, 0.0),
    )

    model = TransformerRegressor(
        context_length=n_dim,
        n_patches=8,
        d_model=16,
        n_heads=2,
        n_layers=4,
        feedforward_dim=16,
        head_dim=8,
        dropout=0.1,
        use_fft=True,
    )
    scaffold = RegressionScaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="mse",
    )

    train_info: RegressionTrainInfo = scaffold.train(
        epochs=300,
        batch_size=64,
        lr=1e-3,
        final_lr=1e-4,
        early_stopping=(10, 5e-2),
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
