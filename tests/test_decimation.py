import numpy as np

from catasta.transformations import Decimation


def main() -> None:
    x: np.ndarray = np.arange(0, 100)
    decimation_factor: int = 2
    decimation: Decimation = Decimation(decimation_factor=decimation_factor)
    y: np.ndarray = decimation(x=x)

    assert np.allclose(y, np.arange(0, 100, decimation_factor))


if __name__ == "__main__":
    main()
