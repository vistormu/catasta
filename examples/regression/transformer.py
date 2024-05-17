from catasta import Scaffold, CatastaDataset
from catasta.models import TransformerRegressor
from catasta.transformations import (
    Normalization,
    WindowSliding,
    Slicing,
    Custom,
)
from catasta.dataclasses import EvalInfo


def main() -> None:
    n_dim: int = 768
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

    model = TransformerRegressor(
        context_length=n_dim,
        n_patches=8,
        d_model=8,
        n_heads=4,
        n_layers=2,
        feedforward_dim=4,
        head_dim=4,
        dropout=0.0,
    )
    scaffold = Scaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="mse",
    )

    scaffold.train(
        epochs=10,
        batch_size=256,
        lr=1e-3,
        final_lr=1e-4,
        early_stopping=True,
        data_loader_workers=4,
    )

    info: EvalInfo = scaffold.evaluate()
    print(info)


if __name__ == '__main__':
    main()
