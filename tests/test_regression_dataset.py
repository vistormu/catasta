import numpy as np

from torch.utils.data import DataLoader

from catasta.datasets import RegressionDataset
from catasta.transformations import (
    Custom,
    WindowSliding,
    Slicing,
)


def main() -> None:
    for n_dim in range(4, 13):
        input_transformations = [
            Custom(lambda x: x[7:904]),
            WindowSliding(window_size=n_dim, stride=1),
        ]
        output_transformations = [
            Custom(lambda x: x[7:904]),
            Slicing(amount=n_dim-1, end="left"),
        ]

        dataset = RegressionDataset(
            root="tests/data/test_data/",
            input_transformations=input_transformations,  # type: ignore
            output_transformations=output_transformations,  # type: ignore
            splits=(0.8, 0.1, 0.1),
        )

        train_dataloader = DataLoader(dataset.train, batch_size=3, shuffle=True)

        for x, y in train_dataloader:
            if x.mean() != y.mean():
                print("The values of x and y are not equal!")
                print(f"n_dim: {n_dim}")
                print(np.array(x), np.array(y))
                return

            if y.mean() >= 8:
                print("The value of y is too high!")
                print(f"n_dim: {n_dim}")
                print(np.array(x), np.array(y))
                return

        print("First test passed!")

        for x, y in dataset.validation:  # type: ignore
            if x.mean() != y.mean():
                print("The values of x and y are not equal!")
                print(f"n_dim: {n_dim}")
                print(np.array(x), np.array(y))
                return

            if y.mean() != 8:
                print("The value of y is not 8!")
                print(f"n_dim: {n_dim}")
                print(np.array(x), np.array(y))
                return

        print("Second test passed!")

        for x, y in dataset.test:  # type: ignore
            if x.mean() != y.mean():
                print("The values of x and y are not equal!")
                print(f"n_dim: {n_dim}")
                print(np.array(x), np.array(y))
                return

            if y.mean() != 9:
                print("The value of y is not 9!")
                print(f"n_dim: {n_dim}")
                print(np.array(x), np.array(y))
                return

        print("Third test passed!")


if __name__ == '__main__':
    main()
