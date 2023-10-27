import os
import pandas as pd
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset


def sliding_window(x: np.ndarray, n_dim: int) -> np.ndarray:
    inputs_tiled: np.ndarray = np.tile(x, (n_dim, 1)).T

    for i in range(n_dim):
        inputs_tiled[:, i] = np.roll(inputs_tiled[:, i], -i)

    return inputs_tiled[:-n_dim+1, :]


class ModelDataset(Dataset):
    '''
    `ModelDataset` is a dataset that loads data from a directory of CSV files. Each CSV file must have two columns: `input` and `output`.
    '''

    def __init__(self, *,
                 root: str,
                 n_dim: int = 1,
                 ) -> None:
        '''
        Parameters
        ----------
        root : str
            The root directory of the dataset.
        n_dim : int, optional
            The number of dimensions of the input data. If `n_dim` is greater than 1, the input data will be converted to a sliding window of size `n_dim`.

        Raises
        ------
        ValueError
            If `root` is not a directory.
        ValueError
            If `n_dim` is less than 1.
        ValueError
            If any CSV file in `root` does not have the columns `input` and `output`.
        ValueError
            If any CSV file in `root` does not have the same number of rows for `input` and `output`.
        ValueError
            If `root` does not contain at least one CSV file.

        Examples
        --------
        >>> dataset: ModelDataset = ModelDataset(root="data/", n_dim=4)
        '''

        if not os.path.isdir(root):
            raise ValueError(f"root must be a directory")
        if n_dim < 1:
            raise ValueError(f"n_dim must be greater than 0")

        inputs: list[np.ndarray] = []
        outputs: list[np.ndarray] = []
        filename_counter: int = 0
        for filename in os.listdir(root):
            if not filename.endswith(".csv"):
                continue
            filename_counter += 1

            data_frame: pd.DataFrame = pd.read_csv(root + filename)

            if 'input' not in data_frame.columns or 'output' not in data_frame.columns:
                raise ValueError(f"CSV file {filename} must have the columns 'input' and 'output'")

            input: np.ndarray = data_frame['input'].to_numpy().flatten()
            output: np.ndarray = data_frame['output'].to_numpy().flatten()

            if len(input) != len(output):
                raise ValueError(f"CSV file {filename} must have the same number of rows for 'input' and 'output'")

            inputs.append(input.reshape(-1, 1) if n_dim == 1 else sliding_window(input, n_dim))
            outputs.append(output if n_dim == 1 else output[-inputs[-1].shape[0]:])

        if filename_counter == 0:
            raise ValueError(f"Directory {root} must contain at least one CSV file")

        self.inputs: np.ndarray = np.concatenate(inputs)
        self.outputs: np.ndarray = np.concatenate(outputs)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return torch.tensor(self.inputs[index]), torch.tensor(self.outputs[index])
