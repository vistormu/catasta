import numpy as np
import matplotlib.pyplot as plt

from catasta.models import FeedforwardRegressor
from catasta.datasets import ModelDataset
from catasta.scaffolds import RegressorScaffold
from catasta.entities import EvalInfo


def main() -> None:
    n_dim: int = 20
    dataset: ModelDataset = ModelDataset(root="tests/data/nylon_strain/", n_dim=n_dim)
    # dataset: ModelDataset = ModelDataset(root="tests/data/nylon_stress/", n_dim=n_dim)

    model: FeedforwardRegressor = FeedforwardRegressor(input_dim=n_dim,
                                                       hidden_dims=[64, 32, 16],
                                                       output_dim=1,
                                                       dropout=0.1,
                                                       )

    scaffold: RegressorScaffold = RegressorScaffold(model, dataset)

    losses: np.ndarray = scaffold.train(epochs=100,
                                        batch_size=256,
                                        train_split=6/7,
                                        lr=1e-3,
                                        )

    # plt.figure(figsize=(30, 20))
    # plt.plot(losses)
    # plt.show()

    info: EvalInfo = scaffold.evaluate()

    # input = info.input[:, -1].flatten()[1500:1725]
    # real = info.real[1500:1725]
    # predicted = info.predicted[1500:1725]
    # stds = info.stds[1500:1725] if info.stds is not None else None

    # plt.figure(figsize=(10, 10))
    # plt.plot(input, real, label="true", color="#2f2f2f")
    # plt.plot(input, predicted, label="predicted", color="#0B7AD5")
    # if stds is not None:
    #     plt.fill_between(predicted,
    #                      predicted - stds,
    #                      predicted + stds,
    #                      color="#90caf9",
    #                      alpha=0.75,
    #                      label="std",
    #                      )
    # plt.legend()
    # plt.savefig("tests/figures/stress_hysteresis_ff.svg")

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
    # plt.savefig("tests/figures/stress_plot_ff.svg")

    print(info)


if __name__ == "__main__":
    main()
