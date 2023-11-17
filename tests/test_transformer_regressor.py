import numpy as np
import matplotlib.pyplot as plt

from catasta.models import TransformerRegressor
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.entities import RegressionEvalInfo, RegressionTrainInfo

from vclog import Logger


def main() -> None:
    n_dim: int = 64
    dataset = RegressionDataset(
        # root="tests/data/nylon_carmen/strain/",
        root="tests/data/wire_lisbeth/strain/",
        # root="tests/data/steps/",
        context_length=n_dim,
        prediction_length=1,
        splits=(6/7, 1/7, 0.0),
    )
    # 256, //4, 1, 64, 2, 2, 128
    # 256, //4, 1, 16, 2, 2, 32
    model = TransformerRegressor(
        context_length=n_dim,
        patch_size=n_dim // 4,
        output_dim=1,
        d_model=16,
        n_heads=2,
        n_layers=2,
        feedforward_dim=32,
    )
    scaffold = RegressionScaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="smooth_l1",
    )

    train_info: RegressionTrainInfo = scaffold.train(
        epochs=200,
        batch_size=32,
        lr=1e-3,
    )

    plt.figure(figsize=(30, 20))
    plt.plot(train_info.train_loss, label="train loss", color="black")
    plt.plot(train_info.eval_loss, label="eval loss", color="red")
    plt.legend()
    plt.show()

    Logger.debug(f"min train loss: {np.min(train_info.train_loss):.4f}, "
                 f"min eval loss: {np.min(train_info.eval_loss):.4f}")  # type: ignore

    info: RegressionEvalInfo = scaffold.evaluate()

    plt.figure(figsize=(30, 20))
    plt.plot(info.real, label="real", color="black")
    plt.plot(info.predicted, label="predictions", color="red")
    plt.legend()
    plt.show()

    Logger.debug(info)


if __name__ == '__main__':
    main()
