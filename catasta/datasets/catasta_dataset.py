import os
from typing import NamedTuple, Literal

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset

from vclog import Logger

from ..transformations import Transformation


def scan_classes(dir: str) -> list[str]:
    classes: list[str] = []
    for entry in os.scandir(dir):
        if entry.is_dir():
            classes.append(entry.name)

    classes.sort()

    return classes


def scan_splits(root: str) -> tuple[str, str, str]:
    train_dir_name: str = ""
    validation_dir_name: str = ""
    test_dir_name: str = ""

    splits: list[str] = scan_classes(root)
    for split in splits:
        if "train" in split:
            train_dir_name = split
        elif "val" in split:
            validation_dir_name = split
        elif "test" in split:
            test_dir_name = split

    if not train_dir_name:
        raise ValueError(f"a directory that contains the subword 'train' must be present in {root}")
    elif not validation_dir_name and not test_dir_name:
        raise ValueError(f"at least a validation or test split must be present in {root}")
    elif not validation_dir_name and test_dir_name:
        Logger("catasta").warning(f"no validation split found in {root}. Using test split as validation split.")
        validation_dir_name = test_dir_name
    elif not test_dir_name and validation_dir_name:
        Logger("catasta").warning(f"no test split found in {root}. Using validation split as test split.")
        test_dir_name = validation_dir_name

    return train_dir_name, validation_dir_name, test_dir_name


class CatastaDataset:
    """Class to handle Catasta datasets. Catasta datasets are organized in a directory with the following structure:
    root
    ├── train
    ├── val
    └── test
    Each of the train, val, and test directories contain CSV or image files. For regression tasks, the CSV files must have the columns 'input' and 'output'.

    Attributes
    ----------
    root : str
        The root directory of the Catasta dataset.
    task : Literal["regression", "classification"]
        The task of the dataset.
    train : Dataset
        The training subset of the dataset.
    validation : Dataset
        The validation subset of the dataset.
    test : Dataset
        The test subset of the dataset.
    """

    def __init__(self,
                 root: str,
                 task: Literal["regression", "classification"],
                 input_transformations: list[Transformation] = [],
                 output_transformations: list[Transformation] = [],
                 input_name: str | list[str] = "input",
                 output_name: str = "output",
                 ) -> None:
        """Initialize the CatastaDataset object.

        Parameters
        ----------
        root : str_
            The root directory of the Catasta dataset.
        task : Literal["regression", "classification"]
            The task of the dataset.
        input_transformations : list[~catasta.transformations.Transformation], optional
            A list of transformations to apply to the input data, by default [].
        output_transformations : list[~catasta.transformations.Transformation], optional
            A list of transformations to apply to the output data, by default [].
        input_name : str | list[str], optional
            The name of the columns in the CSV files that contains the input data, by default "input".
        output_name : str, optional
            The name of the column in the CSV files that contains the output data, by default "output".

        Raises
        ------
        ValueError
            If the root directory does not exist.
            If the task is not 'regression' or 'classification'.
            If the train split is not found.
            If the validation split is not found and the test split is not found.
            If the directories do not contain at least one CSV file (regression).
            If the number of classes in the train and validation splits are not the same (classification).
            If the number of classes in the train and test splits are not the same (classification).
        """
        self.task: Literal["regression", "classification"] = task
        self.root: str = root if root.endswith("/") else root + "/"
        if not os.path.isdir(self.root):
            raise ValueError(f"root must be a directory. Found {self.root}")

        splits: tuple[str, str, str] = scan_splits(self.root)

        if self.task == "regression":
            self.train: Dataset = RegressionSubset(self.root + splits[0],
                                                   input_transformations,
                                                   output_transformations,
                                                   input_name,
                                                   output_name,
                                                   )
            self.validation: Dataset = RegressionSubset(self.root + splits[1],
                                                        input_transformations,
                                                        output_transformations,
                                                        input_name,
                                                        output_name,
                                                        )
            self.test: Dataset = RegressionSubset(self.root + splits[2],
                                                  input_transformations,
                                                  output_transformations,
                                                  input_name,
                                                  output_name,
                                                  )
        elif self.task == "classification":
            self.train: Dataset = ClassificationSubset(self.root + splits[0],
                                                       input_transformations,
                                                       )
            self.validation: Dataset = ClassificationSubset(self.root + splits[1],
                                                            input_transformations,
                                                            )
            self.test: Dataset = ClassificationSubset(self.root + splits[2],
                                                      input_transformations,
                                                      )

            # same number of classes in all splits
            if len(self.train.classes) == 0:
                raise ValueError(f"no classes found in {self.train.path}")

            if len(self.train.classes) != len(self.validation.classes):
                raise ValueError(f"number of classes in train and validation splits must be the same")

            if len(self.train.classes) != len(self.test.classes):
                raise ValueError(f"number of classes in train and test splits must be the same")

        else:
            raise ValueError(f"task must be either 'regression' or 'classification'. Found {task}")


class RegressionSubset(Dataset):
    def __init__(self,
                 path: str,
                 input_transformations: list[Transformation],
                 output_transformations: list[Transformation],
                 input_name: str | list[str],
                 output_name: str,
                 ) -> None:
        self.root: str = path if path.endswith("/") else path + "/"

        self.input_transformations: list[Transformation] = input_transformations
        self.output_transformations: list[Transformation] = output_transformations
        self.inputs, self.outputs = self._get_data(input_name, output_name)

    def _get_data(self, input_name: str | list[str], output_name: str) -> tuple[np.ndarray, np.ndarray]:
        inputs: list[np.ndarray] = []
        outputs: list[np.ndarray] = []

        filenames: list[str] = list(filter(lambda x: x.endswith(".csv"), os.listdir(self.root)))

        if not filenames:
            raise ValueError(f"Directory {self.root} must contain at least one CSV file")

        inputs: list[np.ndarray] = []
        outputs: list[np.ndarray] = []
        for filename in filenames:
            df: pd.DataFrame = pd.read_csv(self.root + filename)

            input = df[input_name].to_numpy()
            output = df[output_name].to_numpy()

            for t in self.input_transformations:
                input = t(input)
            for t in self.output_transformations:
                output = t(output)

            inputs.append(input)
            outputs.append(output)

        return np.concatenate(inputs), np.concatenate(outputs)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return torch.tensor(self.inputs[index]).view(-1), torch.tensor(self.outputs[index]).squeeze()


EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ".csv")


class Sample(NamedTuple):
    path: str
    label: str


class ClassificationSubset(Dataset):
    def __init__(self,
                 path: str,
                 input_transformations: list[Transformation],
                 ) -> None:
        self.path: str = path if path.endswith("/") else path + "/"

        self.input_transformations: list[Transformation] = input_transformations

        self.classes: list[str] = scan_classes(path)
        self.samples: list[Sample] = []
        for class_name in self.classes:
            class_path: str = os.path.join(path, class_name)

            # continue if a file is found
            if not os.path.isdir(class_path):
                continue

            # get all images paths and corresponding class names
            for root, _, files in os.walk(class_path):
                for file in files:
                    if not file.lower().endswith(EXTENSIONS):
                        continue

                    sample = Sample(os.path.join(root, file), class_name)
                    self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample: Sample = self.samples[index]

        if sample.path.endswith(".csv"):
            sample_array: np.ndarray = pd.read_csv(sample.path).to_numpy()
        else:
            with open(sample.path, "rb") as f:
                img: Image.Image = Image.open(f)
                img = img.convert("RGB")

            sample_array: np.ndarray = np.array(img)

        for transformation in self.input_transformations:
            sample_array = transformation(sample_array)

        label: int = self.classes.index(sample.label)

        return torch.tensor(sample_array), label
