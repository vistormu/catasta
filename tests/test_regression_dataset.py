import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from catasta.datasets import RegressionDataset


def main() -> None:
    inputs_1: np.ndarray = np.arange(5)
    outputs_1: np.ndarray = np.arange(5)
    inputs_2: np.ndarray = np.arange(4, 10)
    outputs_2: np.ndarray = np.arange(4, 10)

    pd.DataFrame({
        'input': inputs_1,
        'output': outputs_1
    }).to_csv("tests/test_1.csv", index=False)

    pd.DataFrame({
        'input': inputs_2,
        'output': outputs_2
    }).to_csv("tests/test_2.csv", index=False)

    context_length: int = 3
    prediction_length: int = 2

    dataset = RegressionDataset(root="tests/", context_length=context_length, prediction_length=prediction_length)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch_x, batch_y in data_loader:
        print(batch_x, batch_y)

    os.remove("tests/test_1.csv")
    os.remove("tests/test_2.csv")


if __name__ == '__main__':
    main()
