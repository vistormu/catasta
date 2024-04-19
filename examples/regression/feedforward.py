from catasta import Scaffold, CatastaDataset
from catasta.models import FeedforwardRegressor
from catasta.transformations import (
    Normalization,
    WindowSliding,
    Slicing,
    Custom,
)

from vclog import Logger


def main() -> None:
    n_dim: int = 128
    dataset_root: str = "data/nylon_wire/"
    input_trasnsformations = [
        Custom(lambda x: x[:10_000]),
        Normalization("minmax"),
        WindowSliding(window_size=n_dim, stride=1),
    ]
    output_trasnsformations = [
        Custom(lambda x: x[:10_000]),
        Normalization("minmax"),
        Slicing(amount=n_dim - 1, end="left"),
    ]
    dataset = CatastaDataset(
        root=dataset_root,
        task="regression",
        input_transformations=input_trasnsformations,
        output_transformations=output_trasnsformations,
    )

    model = FeedforwardRegressor(
        context_length=n_dim,
        hidden_dims=[8, 16, 8],
        dropout=0.0,
        use_layer_norm=True,
        activation="relu",
    )
    scaffold = Scaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="mse",
    )

    scaffold.train(
        epochs=100,
        batch_size=256,
        lr=1e-3,
        early_stopping=True,
    )

    info = scaffold.evaluate()
    Logger.debug(info)


if __name__ == '__main__':
    main()
