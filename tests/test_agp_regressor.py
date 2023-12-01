import numpy as np

from catasta.models import ApproximateGPRegressor
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.dataclasses import RegressionEvalInfo, RegressionTrainInfo
from catasta.transformations import Normalization, Decimation, WindowSliding, FIRFiletring, Slicing
from catasta.utils import Plotter

from vclog import Logger


def main() -> None:
    n_inducing_points: int = 16
    n_dim: int = 128

    input_trasnsformations = [
        Slicing(amount=1, end="right"),
        Normalization("zscore"),
        Decimation(decimation_factor=100),
        FIRFiletring(fs=100, cutoff=0.1, numtaps=101),
        WindowSliding(window_size=n_dim, stride=1),
    ]
    output_trasnsformations = [
        Slicing(amount=1, end="right"),
        Normalization("zscore"),
        Decimation(decimation_factor=100),
        FIRFiletring(fs=100, cutoff=0.1, numtaps=101),
        Slicing(amount=n_dim - 1, end="left"),
    ]

    # dataset_root: str = "tests/data/steps/"
    dataset_root: str = "tests/data/nylon_carmen/paper/strain/mixed_10_20/"
    # dataset_root: str = "tests/data/wire_lisbeth/strain/"
    dataset = RegressionDataset(
        root=dataset_root,
        input_transformations=input_trasnsformations,
        output_transformations=output_trasnsformations,
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
        dataset=dataset,
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
