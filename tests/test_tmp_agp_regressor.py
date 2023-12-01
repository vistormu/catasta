import numpy as np

from catasta.models import ApproximateGPRegressor
from tests.tmp_regressor_dataset import TMPRegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.dataclasses import RegressionEvalInfo, RegressionTrainInfo
from catasta.utils import Plotter

from vclog import Logger


def main() -> None:
    n_inducing_points: int = 128
    n_dim: int = 20
    dataset_root: str = "tests/data/nylon_carmen/strain/"
    dataset = TMPRegressionDataset(
        root=dataset_root,
        context_length=n_dim,
        prediction_length=1,
        splits=(6/7, 0.0, 1/7),
    )
    model = ApproximateGPRegressor(
        n_inducing_points=n_inducing_points,
        n_inputs=n_dim,
        kernel="rq",
        mean="zero"
    )
    scaffold = RegressionScaffold(
        model=model,
        dataset=dataset,  # type: ignore
        optimizer="adamw",
        loss_function="variational_elbo",
    )

    train_info: RegressionTrainInfo = scaffold.train(
        epochs=100,
        batch_size=128,
        lr=1e-2,
        final_lr=1e-3,
        early_stopping=True,
    )
    Logger.debug(f"min train loss: {np.min(train_info.train_loss):.4f}")

    info: RegressionEvalInfo = scaffold.evaluate()
    Logger.debug(info)

    plotter = Plotter(
        train_info=train_info,
        eval_info=info,
    )
    plotter.plot_all()


if __name__ == '__main__':
    main()
