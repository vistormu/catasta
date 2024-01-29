import numpy as np

from catasta.transformations import WindowSliding, Slicing


def main() -> None:
    x: np.ndarray = np.arange(0, 100)
    window_size: int = 10
    stride: int = 1
    window_sliding: WindowSliding = WindowSliding(
        window_size=window_size, stride=stride
    )
    y: np.ndarray = window_sliding(x)
    slicing: Slicing = Slicing(amount=window_size - 1, end="left")

    x = slicing(x)

    for x, y in zip(x, y):
        print(y, x)


if __name__ == "__main__":
    main()
