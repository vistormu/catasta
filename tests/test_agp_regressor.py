import numpy as np
import matplotlib.pyplot as plt

from catasta.models import ApproximateGPRegressor
from catasta.datasets import ModelDataset
from catasta.scaffolds import GaussianRegressorScaffold
from catasta.entities import EvalInfo


def main() -> None:
    n_inducing_points: int = 64
    n_dim: int = 20
    dataset = ModelDataset(root="tests/data/nylon_strain/", n_dim=n_dim)
    model = ApproximateGPRegressor(n_inducing_points, n_dim)
    scaffold = GaussianRegressorScaffold(model, dataset)

    losses: np.ndarray = scaffold.train(
        epochs=100,
        batch_size=256,
        train_split=0.8571,
        lr=1e-3,
        stop_condition=1e-1,
    )

    plt.figure(figsize=(30, 20))
    plt.plot(losses)
    plt.show()

    info: EvalInfo = scaffold.evaluate(plot_results=True)

    print(info)


if __name__ == '__main__':
    main()
