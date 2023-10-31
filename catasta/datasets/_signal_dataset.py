import os
import pandas as pd
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SignalDataset(Dataset):
    '''
    The `SignalDataset` class is a subclass of `torch.utils.data.Dataset` that loads a dataset of signals from a directory. It returns random subsequences of a fixed length from the dataset. The directories should be the labels of the target. The subsequences are returned as a tuple of two tensors: the first tensor contains the input data, and the second tensor contains the target data.
    '''

    def __init__(self, *,
                 root: str,
                 sequence_length: int,
                 ) -> None:
        '''
        Parameters
        ----------
        root : str
            The root directory of the dataset. It must be normalized in the interval [0, 1].
        sequence_length : int
            The length of the subsequences.
        '''
        # Variables
        self.sequence_length: int = sequence_length
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get data
        data_files: list[str] = [os.path.join(root, file) for root, _, files in os.walk(root) for file in files if file.endswith(".csv")]
        if not data_files:
            raise FileNotFoundError(f'Could not find any data file in {root}.')

        self.multi_data: np.ndarray = np.array([pd.read_csv(file).to_numpy().flatten() for file in data_files])
        if np.max(self.multi_data) > 1 or np.min(self.multi_data) < 0:
            raise ValueError(f'The data in {root} is not normalized.')

        self.multi_data = (self.multi_data*(self.sequence_length-1)).astype(int)

        # Get targets
        self.classes: list[str] = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
        self.n_classes: int = len(self.classes)
        if not self.classes:
            raise FileNotFoundError(f'Could not find any class folder in {root}.')

        self.class_to_idx: dict[str, int] = {class_name: index for index, class_name in enumerate(self.classes)}

        self.targets: np.ndarray = np.array([self.class_to_idx[os.path.basename(os.path.dirname(file))] for file in data_files])

    def __len__(self) -> int:
        return self.multi_data.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sequence: Tensor = torch.tensor(self.multi_data[index], device=self.device)
        target: Tensor = torch.tensor(self.targets[index], device=self.device)

        return (sequence, target)
