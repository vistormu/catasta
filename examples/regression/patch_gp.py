from catasta import Scaffold, CatastaDataset
from catasta.models import PatchGPRegressor
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

    model = PatchGPRegressor(
        n_inducing_points=32,
        context_length=n_dim,
        n_patches=4,
        d_model=n_dim // 4,
        kernel="rq",
        mean="constant",
        pooling="mean",
        use_ard=True,
    )

    scaffold = Scaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="variational_elbo",
    )

    scaffold.train(
        epochs=100,
        batch_size=256,
        lr=1e-3,
        data_loader_workers=4,
    )

    info: EvalInfo = scaffold.evaluate()
    print(info)


if __name__ == '__main__':
    main()
