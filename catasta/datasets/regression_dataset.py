import os
import pandas as pd
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset

from ..transformations import Transformation


class RegressionDataset(Dataset):
    def __init__(self, *,
                 root: str,
                 input_transformations: list[Transformation] = [],
                 output_transformations: list[Transformation] = [],
                 splits: tuple[float, float, float] = (0.8, 0.1, 0.1),
                 ) -> None:
        self.root: str = root if root.endswith("/") else root + "/"
        self.train_split: float = splits[0]
        self.validation_split: float = splits[1]
        self.test_split: float = splits[2]

        self.input_transformations: list[Transformation] = input_transformations
        self.output_transformations: list[Transformation] = output_transformations

        self.inputs: np.ndarray = np.array([])
        self.outputs: np.ndarray = np.array([])

        self._check_arguments()
        self.inputs, self.outputs = self._prepare_data(*self._get_data())
        self.train, self.validation, self.test = self._split_dataset()

    def _check_arguments(self) -> None:
        if not os.path.isdir(self.root):
            raise ValueError(f"root must be a directory. Found {self.root}")
        splits_sum: float = round(sum([self.train_split, self.validation_split, self.test_split]), 4)
        if splits_sum != 1:
            raise ValueError(f"splits must sum to 1. Found {splits_sum}")
        if self.validation_split == 0 and self.test_split == 0:
            raise ValueError(f"at least a validation or test split must be greater than 0")
        if self.train_split == 0:
            raise ValueError(f"train split must be greater than 0")

    def _get_data(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        inputs: list[np.ndarray] = []
        outputs: list[np.ndarray] = []

        filename_counter: int = 0
        filenames: list[str] = os.listdir(self.root)
        filenames.sort()
        for filename in filenames:
            if not filename.endswith(".csv"):
                continue
            filename_counter += 1

            data_frame: pd.DataFrame = pd.read_csv(self.root + filename)

            if 'input' not in data_frame.columns or 'output' not in data_frame.columns:
                raise ValueError(f"CSV file {filename} must have the columns 'input' and 'output'")

            inputs.append(data_frame['input'].to_numpy().flatten())
            outputs.append(data_frame['output'].to_numpy().flatten())

            if len(inputs[-1]) != len(outputs[-1]):
                raise ValueError(f"CSV file {filename} must have the same number of rows for 'input' and 'output'")

        if filename_counter == 0:
            raise ValueError(f"Directory {self.root} must contain at least one CSV file")

        return inputs, outputs

    def _prepare_data(self, inputs: list[np.ndarray], outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        transformed_inputs: list[np.ndarray] = []
        transformed_outputs: list[np.ndarray] = []

        for input, output in zip(inputs, outputs):
            for transformation in self.input_transformations:
                input = transformation(input)
            for transformation in self.output_transformations:
                output = transformation(output)

            transformed_inputs.append(input)
            transformed_outputs.append(output)

        inputs_array: np.ndarray = np.concatenate(transformed_inputs)
        outputs_array: np.ndarray = np.concatenate(transformed_outputs)

        return inputs_array, outputs_array

    def _split_dataset(self) -> tuple[Subset, Subset | None, Subset | None]:
        train_index: int = int(self.train_split * len(self))
        validation_index: int = train_index + int(self.validation_split * len(self))
        test_index: int = validation_index + int(self.test_split * len(self))
        train: Subset = Subset(self, range(train_index))
        validation: Subset | None = Subset(self, range(train_index, validation_index)) if self.validation_split > 0 else None
        test: Subset | None = Subset(self, range(validation_index, test_index)) if self.test_split > 0 else None

        return train, validation, test

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return torch.tensor(self.inputs[index]), torch.tensor(self.outputs[index]).squeeze()
