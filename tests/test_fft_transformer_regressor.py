import numpy as np
import matplotlib.pyplot as plt

from catasta.models import FFTTransformerRegressor
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.dataclasses import RegressionEvalInfo, RegressionTrainInfo
from catasta.transformations import Normalization, Decimation, WindowSliding, FIRFiletring, Slicing
from catasta.utils import Plotter

from vclog import Logger


def main() -> None:
    n_dim: int = 256
    input_trasnsformations = [
        Slicing(amount=1, end="right"),
        Normalization("zscore"),
        Decimation(decimation_factor=100),
        FIRFiletring(fs=100, cutoff=0.1, numtaps=101),
        WindowSliding(window_size=n_dim, stride=1),
    ]
    output_trasnsformations = [
        Slicing(amount=1, end="right"),
        Normalization("zscore"),
        Decimation(decimation_factor=100),
        FIRFiletring(fs=100, cutoff=0.1, numtaps=101),
        Slicing(amount=n_dim - 1, end="left"),
    ]
    dataset = RegressionDataset(
        root="tests/data/nylon_carmen/paper/strain/mixed_10_20/",
        # root="tests/data/wire_lisbeth/strain/",
        # root="tests/data/steps/",
        input_transformations=input_trasnsformations,
        output_transformations=output_trasnsformations,
        splits=(6/7, 1/7, 0.0),
    )
    # 1024, 4, 1, 16, 2, 2, 32, 4
    # 512, 4, 1, 64, 2, 2, 32, 4
    model = FFTTransformerRegressor(
        context_length=n_dim,
        n_patches=8,
        output_dim=1,
        d_model=64,
        n_heads=4,
        n_layers=4,
        feedforward_dim=128,
        head_dim=64,
    )
    scaffold = RegressionScaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="huber",
    )

    # 200, 64, 1e-3, 5e-4
    # 100, 128, 1e-3, 5e-4
    train_info: RegressionTrainInfo = scaffold.train(
        epochs=500,
        batch_size=128,
        lr=1e-3,
        final_lr=1e-4,
        early_stopping=True,
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
