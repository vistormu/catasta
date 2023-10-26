import pandas as pd

from catasta.models import ApproximateGPRegressor
from catasta.datasets import ModelDataset
from catasta.scaffolds import GaussianRegressorScaffold
from catasta.entities import EvalInfo

from vclog import Logger


def main(data: str, signal: str) -> None:
    dataset_root: str = "tests/data/paper/" + data + "_" + signal + "/"
    Logger.info(f"Loading dataset from {dataset_root}")

    n_dim: int = 20
    n_inducing_points: int = 128
    dataset = ModelDataset(root=dataset_root, n_dim=n_dim)

    model = ApproximateGPRegressor(n_inducing_points, n_dim, kernel="rq", mean="constant")
    scaffold = GaussianRegressorScaffold(model, dataset)

    scaffold.train(
        epochs=100,
        batch_size=256,
        train_split=6/7,
        lr=1e-3,
    )

    Logger.info("Evaluating...")

    info: EvalInfo = scaffold.evaluate()

    data_to_save: dict[str, float] = {
        "rmse%": round(info.rmse*100, 2),
        "r2": round(info.r2, 4),
    }

    filename: str = "tests/results/" + data + "_" + signal + ".csv"
    pd.DataFrame(data_to_save, index=[0]).to_csv(filename, index=False)

    Logger.info(f"Saved results to {filename}")


if __name__ == '__main__':
    signals: list[str] = ["sin_20", "sin_30", "sin_40", "mixed_10_20", "mixed_15_25", "mixed_20_30", "triangular"]
    datas: list[str] = ["strain", "stress"]

    for data in datas:
        for signal in signals:
            main(data, signal)
