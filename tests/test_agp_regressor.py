import numpy as np

from catasta.models import ApproximateGPRegressor
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.entities import RegressionEvalInfo, RegressionTrainInfo

from vclog import Logger


def main() -> None:
    n_inducing_points: int = 128
    n_dim: int = 20
    dataset = RegressionDataset(root="tests/data/steps/", n_dim=n_dim)
    model = ApproximateGPRegressor(n_inducing_points, n_dim, kernel="rq", mean="zero")

    scaffold = RegressionScaffold(
        model=model,
        dataset=dataset,
    )

    train_info: RegressionTrainInfo = scaffold.train(
        epochs=100,
        batch_size=256,
        train_split=6/7,
        lr=1e-3,
    )

    Logger.debug(f"min train loss: {np.min(train_info.train_loss):.4f}")

    info: RegressionEvalInfo = scaffold.evaluate()

    Logger.debug(info)


if __name__ == '__main__':
    main()
