import numpy as np
import matplotlib.pyplot as plt

from catasta.models import ApproximateGPRegressor
from catasta.datasets import ModelDataset
from catasta.scaffolds import GaussianRegressorScaffold
from catasta.entities import EvalInfo


def main() -> None:
    n_inducing_points: int = 128
    n_dim: int = 20
    dataset = ModelDataset(root="tests/data/nylon_strain/", n_dim=n_dim)
    # dataset = ModelDataset(root="tests/data/nylon_stress/", n_dim=n_dim)
    model = ApproximateGPRegressor(n_inducing_points, n_dim, kernel="rq", mean="zero")
    scaffold = GaussianRegressorScaffold(model, dataset)

    losses: np.ndarray = scaffold.train(
        epochs=100,
        batch_size=256,
        train_split=6/7,
        lr=1e-3,
    )

    plt.figure(figsize=(30, 20))
    plt.plot(losses)
    plt.show()

    info: EvalInfo = scaffold.evaluate()

    plt.figure(figsize=(30, 20))
    plt.plot(info.predicted, label="true")
    plt.plot(info.real, label="pred")
    if info.stds is not None:
        plt.fill_between(range(len(info.predicted)),
                         info.predicted - info.stds,
                         info.predicted + info.stds,
                         alpha=0.5,
                         label="std",
                         )
    plt.legend()
    plt.show()

    print(info)


if __name__ == '__main__':
    main()
