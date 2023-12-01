import os
import pandas as pd
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset


class TMPRegressionDataset(Dataset):
    def __init__(self, *,
                 root: str,
                 context_length: int = 1,
                 prediction_length: int = 1,
                 splits: tuple[float, float, float] = (0.8, 0.1, 0.1),
                 ) -> None:
        self.root: str = root
        self.context_length: int = context_length
        self.prediction_length: int = prediction_length
        self.splits: tuple[float, float, float] = splits

        self.inputs: np.ndarray = np.array([])
        self.outputs: np.ndarray = np.array([])

        self._check_arguments()
        inputs, outputs = self._get_data()
        self._prepare_data(inputs, outputs)

        train_index: int = int(self.splits[0] * len(self))
        validation_index: int = train_index + int(self.splits[1] * len(self))
        test_index: int = validation_index + int(self.splits[2] * len(self))
        self.train: Subset = Subset(self, range(train_index))
        self.validation: Subset | None = Subset(self, range(train_index, validation_index)) if splits[1] > 0 else None
        self.test: Subset | None = Subset(self, range(validation_index, test_index)) if splits[2] > 0 else None

    def _check_arguments(self) -> None:
        if not os.path.isdir(self.root):
            raise ValueError(f"root must be a directory")
        if self.context_length < 1:
            raise ValueError(f"context_length must be at least 1")
        if self.prediction_length < 1:
            raise ValueError(f"prediction_length must be at least 1")
        if round(sum(self.splits), 4) != 1:
            raise ValueError(f"splits must sum to 1. Found {sum(self.splits)}")
        if self.splits[1] == 0 and self.splits[2] == 0:
            raise ValueError(f"at least a validation or test split must be greater than 0")
        if self.splits[0] == 0:
            raise ValueError(f"train split must be greater than 0")

    def _get_data(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        inputs: list[np.ndarray] = []
        outputs: list[np.ndarray] = []

        filename_counter: int = 0
        for filename in os.listdir(self.root):
            if not filename.endswith(".csv"):
                continue
            filename_counter += 1

            data_frame: pd.DataFrame = pd.read_csv(self.root + filename)

            input_1: np.ndarray = data_frame['input_1'].to_numpy().flatten()
            input_2: np.ndarray = data_frame['input_2'].to_numpy().flatten()
            input_3: np.ndarray = data_frame['input_3'].to_numpy().flatten()
            input_4: np.ndarray = data_frame['input_4'].to_numpy().flatten()
            output: np.ndarray = data_frame['output'].to_numpy().flatten()

            new_input: np.ndarray = np.stack([input_1, input_2, input_3, input_4], axis=1)
            inputs.append(new_input)
            outputs.append(output)

            if len(inputs[-1]) < self.context_length + self.prediction_length:
                raise ValueError(f"CSV file {filename} has {len(inputs[-1])} rows, and the sum of the context and prediction lengths is {self.context_length + self.prediction_length}")

        if filename_counter == 0:
            raise ValueError(f"Directory {self.root} must contain at least one CSV file")

        return inputs, outputs

    def _prepare_data(self, inputs: list[np.ndarray], outputs: list[np.ndarray]) -> None:
        reshaped_inputs: list[np.ndarray] = []
        reshaped_outputs: list[np.ndarray] = []

        for input, output in zip(inputs, outputs):
            reshaped_input: list[np.ndarray] = []
            reshaped_output: list[np.ndarray] = []

            n_iterations: int = input.shape[0] - self.context_length - self.prediction_length + 2
            n: int = self.context_length // input.shape[1]
            for i in range(n_iterations):
                context: np.ndarray = input[i: i+n, :].T.reshape(-1)

                reshaped_input.append(context)

                prediction: np.ndarray = output[i+self.context_length-1: i+self.context_length+self.prediction_length-1]
                reshaped_output.append(prediction)

            reshaped_inputs.append(np.array(reshaped_input))
            reshaped_outputs.append(np.array(reshaped_output))

        self.inputs: np.ndarray = np.concatenate(reshaped_inputs)
        self.outputs: np.ndarray = np.concatenate(reshaped_outputs)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return torch.tensor(self.inputs[index]), torch.tensor(self.outputs[index]).squeeze()
