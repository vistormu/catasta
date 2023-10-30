import numpy as np
import matplotlib.pyplot as plt

from catasta.models import ApproximateGPRegressor
from catasta.datasets import ModelDataset
from catasta.scaffolds import GaussianRegressorScaffold
from catasta.entities import EvalInfo


def main() -> None:
    n_inducing_points: int = 128
    n_dim: int = 20
    # dataset = ModelDataset(root="tests/data/nylon_strain/", n_dim=n_dim)
    dataset = ModelDataset(root="tests/data/nylon_stress/", n_dim=n_dim)
    model = ApproximateGPRegressor(n_inducing_points, n_dim, kernel="rq", mean="zero")
    model.load("tests/models/model.pt")
    scaffold = GaussianRegressorScaffold(model, dataset)
    scaffold.train_split = 6/7

    # losses: np.ndarray = scaffold.train(
    #     epochs=100,
    #     batch_size=256,
    #     train_split=6/7,
    #     lr=1e-3,
    # )

    # model.save("tests/models/model.pt")

    # plt.figure(figsize=(30, 20))
    # plt.plot(losses)
    # plt.show()

    info: EvalInfo = scaffold.evaluate()

    input = info.input[:, -1].flatten()[1500:1725]
    real = info.real[1500:1725]
    predicted = info.predicted[1500:1725]
    stds = info.stds[1500:1725] if info.stds is not None else None

    plt.figure(figsize=(10, 10))
    plt.plot(input, real, label="true", color="#2f2f2f")
    plt.plot(input, predicted, label="predicted", color="#0B7AD5")
    if stds is not None:
        plt.fill_between(input,
                         predicted - stds,
                         predicted + stds,
                         color="#90caf9",
                         alpha=0.75,
                         label="std",
                         )
    plt.legend()
    plt.show()

    # plt.figure(figsize=(10, 10))
    # plt.plot(real, label="true", color="#2f2f2f")
    # plt.plot(predicted, label="predicted", color="#0B7AD5")
    # if stds is not None:
    #     plt.fill_between(range(len(predicted)),
    #                      predicted - stds,
    #                      predicted + stds,
    #                      color="#90caf9",
    #                      alpha=0.75,
    #                      label="std",
    #                      )
    # plt.legend()
    # plt.show()

    # print(info)


if __name__ == '__main__':
    main()
