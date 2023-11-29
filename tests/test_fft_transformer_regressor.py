import numpy as np
import matplotlib.pyplot as plt

from catasta.models import FFTTransformerRegressor
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.dataclasses import RegressionEvalInfo, RegressionTrainInfo
from catasta.utils import Plotter

from vclog import Logger


def main() -> None:
    n_dim: int = 512
    dataset = RegressionDataset(
        # root="tests/data/nylon_carmen/strain/",
        root="tests/data/wire_lisbeth/strain/",
        # root="tests/data/steps/",
        context_length=n_dim,
        prediction_length=1,
        splits=(6/7, 1/7, 0.0),
    )
    # 1024, 4, 1, 16, 2, 2, 32, 4
    # 512, 4, 1, 64, 2, 2, 32, 4
    model = FFTTransformerRegressor(
        context_length=n_dim,
        n_patches=4,
        output_dim=1,
        d_model=64,
        n_heads=2,
        n_layers=2,
        feedforward_dim=32,
        head_dim=4,
    )
    scaffold = RegressionScaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="mse",
    )

    # 200, 64, 1e-3, 5e-4
    # 100, 128, 1e-3, 5e-4
    train_info: RegressionTrainInfo = scaffold.train(
        epochs=200,
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
