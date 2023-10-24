import os
import pandas as pd
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset


class ModelDataset(Dataset):
    '''
    `ModelDataset` is a dataset that loads data from a directory of CSV files. Each CSV file must have two columns: `input` and `output`.
    '''

    def __init__(self, *,
                 root: str,
                 dtype: str = "float32",
                 n_dim: int = 1,
                 ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32 if dtype == "float32" else torch.float64

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

            if n_dim > 1:
                inputs_tiled: np.ndarray = np.tile(input, (n_dim, 1)).T

                for i in range(n_dim):
                    inputs_tiled[:, i] = np.roll(inputs_tiled[:, i], -i)

                new_input: np.ndarray = inputs_tiled[:-n_dim+1, :]
                new_output: np.ndarray = output[-new_input.shape[0]:]

                inputs.append(new_input)
                outputs.append(new_output)

            elif n_dim == 1:
                inputs.append(input)
                outputs.append(output)
            else:
                raise ValueError(f"n_dim must be greater than 0")

        if filename_counter == 0:
            raise ValueError(f"Directory {root} must contain at least one CSV file")

        self.inputs: np.ndarray = np.concatenate(inputs)
        self.outputs: np.ndarray = np.concatenate(outputs)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        input_tensor: Tensor = torch.tensor(self.inputs[index]).to(self.device).to(self.dtype)
        output_tensor: Tensor = torch.tensor(self.outputs[index]).to(self.device).to(self.dtype)

        return input_tensor, output_tensor
