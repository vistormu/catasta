import matplotlib.pyplot as plt

from catasta.models import ApproximateGPRegressor, FeedforwardRegressor
from catasta.datasets import ModelDataset
from catasta.scaffolds import GaussianRegressorScaffold, RegressorScaffold
from catasta.entities import EvalInfo


def main(variable: str, model_id: str) -> None:
    n_inducing_points: int = 128
    n_dim: int = 20
    dataset = ModelDataset(root=f"tests/data/nylon_{variable}/", n_dim=n_dim)

    if model_id == "gp":
        model = ApproximateGPRegressor(n_inducing_points, n_dim, kernel="rq", mean="zero")
    elif model_id == "fnn":
        model = FeedforwardRegressor(input_dim=n_dim,
                                     hidden_dims=[64, 32, 16],
                                     output_dim=1,
                                     dropout=0.1,
                                     )
    else:
        raise ValueError(f"Unknown model: {model_id}")

    if model_id == "gp":
        scaffold = GaussianRegressorScaffold(model, dataset)
    elif model_id == "fnn":
        scaffold = RegressorScaffold(model, dataset)
    else:
        raise ValueError(f"Unknown model: {model_id}")

    scaffold.train(
        epochs=100,
        batch_size=256,
        train_split=6/7,
        lr=1e-3,
    )

    info: EvalInfo = scaffold.evaluate()

    print(info)

    limits = (-2.5, 2.5)
    interval = (500, 1000)

    # input = info.input[:, -1].flatten()[interval[0]:interval[1]]
    real = info.real[interval[0]:interval[1]]
    predicted = info.predicted[interval[0]:interval[1]]
    stds = info.stds[interval[0]:interval[1]] if info.stds is not None else None

    # hysteresis
    # plt.figure(figsize=(10, 10))
    # plt.ylim(*limits)
    # plt.xlim(*limits)
    # plt.plot(input, real, label="true", color="#2f2f2f")
    # plt.plot(input, predicted, label="predicted", color="#0B7AD5")
    # if stds is not None:
    #     plt.fill_between(input,
    #                      predicted - stds,
    #                      predicted + stds,
    #                      color="#90caf9",
    #                      alpha=0.75,
    #                      label="std",
    #                      )
    # plt.legend()
    # plt.savefig(f"tests/figures/{variable}_hysteresis_{model_id}.svg")
    # plt.show()

    # plot
    plt.figure(figsize=(10, 10))
    plt.ylim(*limits)
    plt.plot(real, label="true", color="#2f2f2f")
    plt.plot(predicted, label="predicted", color="#0B7AD5")
    if stds is not None:
        plt.fill_between(range(len(predicted)),
                         predicted - stds,
                         predicted + stds,
                         color="#90caf9",
                         alpha=0.75,
                         label="std",
                         )
    plt.legend()
    plt.savefig(f"tests/figures/{variable}_plot_{model_id}.svg")
    # plt.show()


if __name__ == '__main__':
    for variable in ["strain", "stress"]:
        for model in ["gp", "fnn"]:
            main(variable, model)
