from catasta import Scaffold, CatastaDataset
from catasta.models import ApproximateGPRegressor
from catasta.dataclasses import TrainInfo
from catasta.transformations import (
    Normalization,
    WindowSliding,
    Slicing,
    Custom,
)

from vclog import Logger


def main() -> None:
    n_dim: int = 768
    dataset_root: str = "tests/data/nylon_wire/"
    input_trasnsformations = [
        Custom(lambda x: x[:5_000]),
        Normalization("minmax"),
        WindowSliding(window_size=n_dim, stride=1),
    ]
    output_trasnsformations = [
        Custom(lambda x: x[:5_000]),
        Normalization("minmax"),
        Slicing(amount=n_dim - 1, end="left"),
    ]
    dataset = CatastaDataset(
        root=dataset_root,
        task="regression",
        input_transformations=input_trasnsformations,
        output_transformations=output_trasnsformations,
    )

    model = ApproximateGPRegressor(
        context_length=n_dim,
        n_inducing_points=128,
        kernel="rq",
        mean="constant"
    )
    scaffold = Scaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="variational_elbo",
    )

    train_info: TrainInfo = scaffold.train(
        epochs=100,
        batch_size=256,
        lr=1e-3,
    )
    Logger.debug(f"min train loss: {train_info.best_train_loss:.4f}")

    info = scaffold.evaluate()
    Logger.debug(info)


if __name__ == '__main__':
    main()
