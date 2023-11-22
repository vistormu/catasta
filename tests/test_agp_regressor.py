import numpy as np
import matplotlib.pyplot as plt

from catasta.models import ApproximateGPRegressor
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.dataclasses import RegressionEvalInfo, RegressionTrainInfo

from vclog import Logger


def main() -> None:
    n_inducing_points: int = 128
    n_dim: int = 20
    # dataset_root: str = "tests/data/steps/"
    dataset_root: str = "tests/data/nylon_carmen/strain/"
    dataset = RegressionDataset(
        root=dataset_root,
        context_length=n_dim,
        prediction_length=1,
        splits=(0.9, 0.0, 0.1),
    )
    model = ApproximateGPRegressor(
        n_inducing_points=n_inducing_points,
        n_inputs=n_dim,
        kernel="rq",
        mean="zero"
    )
    scaffold = RegressionScaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="predictive_log",
    )

    train_info: RegressionTrainInfo = scaffold.train(
        epochs=100,
        batch_size=256,
        lr=1e-2,
    )

    Logger.debug(f"min train loss: {np.min(train_info.train_loss):.4f}")

    info: RegressionEvalInfo = scaffold.evaluate()

    plt.figure(figsize=(30, 20))
    plt.plot(info.predicted, label="predictions", color="red")
    plt.plot(info.real, label="real", color="black")
    plt.fill_between(range(len(info.predicted)), info.predicted-1*info.stds, info.predicted+1*info.stds, color="red", alpha=0.2)
    plt.legend()
    plt.show()

    Logger.debug(info)


if __name__ == '__main__':
    main()
