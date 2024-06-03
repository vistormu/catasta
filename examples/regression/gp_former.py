from catasta import Scaffold, CatastaDataset
from catasta.models import GPFormerRegressor
from catasta.transformations import (
    Normalization,
    WindowSliding,
    Slicing,
    Custom,
)
from catasta.dataclasses import EvalInfo
from catasta.utils import set_deterministic


def main() -> None:
    set_deterministic()

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

    model = GPFormerRegressor(
        n_inducing_points=32,
        kernel="rq",
        context_length=n_dim,
        n_patches=8,
        d_model=8,
        n_heads=1,
        n_layers=1,
        feedforward_dim=4,
        head_dim=4,
        dropout=0.5,
        pooling="concat",
        use_ard=True,
        layer_norm=False,
    )
    scaffold = Scaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="variational_elbo",
    )

    scaffold.train(
        epochs=20,
        batch_size=256,
        lr=1e-3,
        data_loader_workers=0,
    )

    info: EvalInfo = scaffold.evaluate()
    print(info)


if __name__ == '__main__':
    main()
