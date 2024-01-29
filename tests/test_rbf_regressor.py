import numpy as np

from catasta.models import RBFRegressor
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.dataclasses import RegressionEvalInfo, RegressionTrainInfo

from vclog import Logger


def main() -> None:
    n_dim: int = 20
    dataset = RegressionDataset(
        # root="tests/data/steps/",
        root="tests/data/nylon_carmen/strain/",
        context_length=n_dim,
        prediction_length=1,
        splits=(0.8, 0.2, 0.0),
    )
    model = RBFRegressor(
        in_features=n_dim,
        out_features=1,
        basis_func="gaussian",
    )
    scaffold = RegressionScaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="huber",
    )

    train_info: RegressionTrainInfo = scaffold.train(
        epochs=200,
        batch_size=64,
        lr=1e-3,
        early_stopping=True,
    )

    Logger.debug(f"min train loss: {np.min(train_info.train_loss):.4f}, "
                 f"min eval loss: {np.min(train_info.eval_loss):.4f}")  # type: ignore

    info: RegressionEvalInfo = scaffold.evaluate()

    Logger.debug(info)


if __name__ == '__main__':
    main()
