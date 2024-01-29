import numpy as np

from torch.utils.data import DataLoader

from catasta.datasets import RegressionDataset
from catasta.transformations import (
    WindowSliding,
    Slicing,
)


def main() -> None:
    n_dim = 5
    input_transformations = [
        WindowSliding(window_size=n_dim, stride=1),
    ]
    output_transformations = [
        Slicing(amount=n_dim-1, end="left"),
    ]

    dataset = RegressionDataset(
        root="tests/data/test_data/",
        input_transformations=input_transformations,  # type: ignore
        output_transformations=output_transformations,  # type: ignore
        splits=(0.8, 0.1, 0.1),
    )

    train_dataloader = DataLoader(dataset.train, batch_size=1, shuffle=True)

    for x, y in train_dataloader:
        assert x.mean() == y.item(), f"mean of x: {x.mean()}, y: {y.item()}"

    for x, y in dataset.test:  # type: ignore
        assert x.mean() == y.item(), f"mean of x: {x.mean()}, y: {y.item()}"

    for x, y in dataset.validation:  # type: ignore
        assert x.mean() == y.item(), f"mean of x: {x.mean()}, y: {y.item()}"


if __name__ == '__main__':
    main()
