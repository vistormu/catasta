import numpy as np
# import matplotlib.pyplot as plt

from catasta.models import ApproximateGPRegressor
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.entities import RegressionEvalInfo, RegressionTrainInfo

from vclog import Logger


def main() -> None:
    n_inducing_points: int = 128
    n_dim: int = 20
    dataset = RegressionDataset(
        root="tests/data/steps/",
        context_length=n_dim,
        splits=(0.8, 0.0, 0.2),
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

    # plt.plot(info.predicted, label="predictions")
    # plt.plot(info.real, label="real")
    # plt.show()

    Logger.debug(info)


if __name__ == '__main__':
    main()
