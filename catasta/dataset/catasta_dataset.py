import os
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..transformations import Transformation


def scan_classes(path: str) -> list[str]:
    classes: list[str] = []
    for entry in os.scandir(path):
        if entry.is_dir():
            classes.append(entry.name)

    classes.sort()
    return classes


def scan_splits(root: str, allow_alias: bool = True) -> tuple[str, str, str]:
    train_dir_name: str = ""
    validation_dir_name: str = ""
    test_dir_name: str = ""

    splits: list[str] = scan_classes(root)
    for split in splits:
        split_lower = split.lower()
        if "train" in split_lower:
            train_dir_name = split
        elif "val" in split_lower:
            validation_dir_name = split
        elif "test" in split_lower:
            test_dir_name = split

    if not train_dir_name:
        raise ValueError(f"a directory that contains the subword 'train' must be present in {root}")

    if not validation_dir_name and not test_dir_name:
        raise ValueError(f"at least a validation or test split must be present in {root}")

    if (not validation_dir_name or not test_dir_name) and not allow_alias:
        raise ValueError("validation and test splits must both be present unless allow_split_alias=True")

    if not validation_dir_name and test_dir_name:
        validation_dir_name = test_dir_name
    elif not test_dir_name and validation_dir_name:
        test_dir_name = validation_dir_name

    return train_dir_name, validation_dir_name, test_dir_name


class CatastaDataset:
    def __init__(
        self,
        root: str,
        task: str,
        input_name: str | list[str] | None = None,
        output_name: str | list[str] | None = None,
        input_transformations: Sequence[Transformation] | None = None,
        output_transformations: Sequence[Transformation] | None = None,
        grayscale: bool = False,
        allow_split_alias: bool = True,
    ) -> None:
        if task in ("regression", "signal_classification") and (input_name is None or output_name is None):
            raise ValueError("input_name and output_name must be provided for regression and signal_classification tasks")

        self.task: str = task
        self.root: str = root
        if not os.path.isdir(self.root):
            raise ValueError(f"root must be a directory. Found {self.root}")

        input_transformations = [] if input_transformations is None else input_transformations
        output_transformations = [] if output_transformations is None else output_transformations

        splits: tuple[str, str, str] = scan_splits(self.root, allow_alias=allow_split_alias)

        if self.task == "regression":
            self.train = RegressionSubset(
                os.path.join(self.root, splits[0]),
                input_name,  # type: ignore[arg-type]
                output_name,  # type: ignore[arg-type]
                input_transformations,
                output_transformations,
            )
            self.validation = RegressionSubset(
                os.path.join(self.root, splits[1]),
                input_name,  # type: ignore[arg-type]
                output_name,  # type: ignore[arg-type]
                input_transformations,
                output_transformations,
            )
            self.test = RegressionSubset(
                os.path.join(self.root, splits[2]),
                input_name,  # type: ignore[arg-type]
                output_name,  # type: ignore[arg-type]
                input_transformations,
                output_transformations,
            )

        elif self.task == "classification":
            self.train = ClassificationSubset(
                os.path.join(self.root, splits[0]),
                input_transformations,
                grayscale,
            )
            self.validation = ClassificationSubset(
                os.path.join(self.root, splits[1]),
                input_transformations,
                grayscale,
            )
            self.test = ClassificationSubset(
                os.path.join(self.root, splits[2]),
                input_transformations,
                grayscale,
            )

            if len(self.train.classes) == 0:
                raise ValueError(f"no classes found in {self.train.path}")
            if len(self.train.classes) != len(self.validation.classes):
                raise ValueError("number of classes in train and validation splits must be the same")
            if len(self.train.classes) != len(self.test.classes):
                raise ValueError("number of classes in train and test splits must be the same")

        elif self.task == "signal_classification":
            self.train = SignalClassificationSubset(
                os.path.join(self.root, splits[0]),
                input_name,  # type: ignore[arg-type]
                output_name,  # type: ignore[arg-type]
                input_transformations,
            )

            self.validation = SignalClassificationSubset(
                os.path.join(self.root, splits[1]),
                input_name,  # type: ignore[arg-type]
                output_name,  # type: ignore[arg-type]
                input_transformations,
                allowed_labels=self.train.known_labels,
            )

            self.test = SignalClassificationSubset(
                os.path.join(self.root, splits[2]),
                input_name,  # type: ignore[arg-type]
                output_name,  # type: ignore[arg-type]
                input_transformations,
                allowed_labels=self.train.known_labels,
            )

            self.classes = [str(label) for label in self.train.known_labels]
        else:
            raise ValueError(
                "task must be either 'regression', 'classification', or 'signal_classification'. "
                f"Found {task}"
            )


class RegressionSubset(Dataset):
    def __init__(
        self,
        path: str,
        input_name: str | list[str],
        output_name: str | list[str],
        input_transformations: Sequence[Transformation],
        output_transformations: Sequence[Transformation],
    ) -> None:
        self.root: str = path
        self.input_transformations = input_transformations
        self.output_transformations = output_transformations
        self.inputs, self.outputs = self._get_data(input_name, output_name)

        self.inputs = torch.tensor(self.inputs)
        self.outputs = torch.tensor(self.outputs)

    def _get_data(self, input_name: str | list[str], output_name: str | list[str]) -> tuple[np.ndarray, np.ndarray]:
        filenames: list[str] = sorted([name for name in os.listdir(self.root) if name.lower().endswith(".csv")])
        if not filenames:
            raise ValueError(f"Directory {self.root} must contain at least one CSV file")

        inputs: list[np.ndarray] = []
        outputs: list[np.ndarray] = []
        for filename in filenames:
            df = pd.read_csv(os.path.join(self.root, filename))
            input_values = df[input_name].to_numpy()
            output_values = df[output_name].to_numpy()

            for transformation in self.input_transformations:
                input_values = transformation(input_values)
            for transformation in self.output_transformations:
                output_values = transformation(output_values)

            inputs.append(np.asarray(input_values))
            outputs.append(np.asarray(output_values))

        return np.concatenate(inputs), np.concatenate(outputs)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        input_tensor: Tensor = self.inputs[index].view(-1)
        if self.outputs.ndim == 1:
            output_tensor = self.outputs[index]
        else:
            output_tensor = self.outputs[index].view(-1)

        return input_tensor, output_tensor


class SignalClassificationSubset(Dataset):
    def __init__(
        self,
        path: str,
        input_name: str | list[str],
        output_name: str | list[str],
        input_transformations: Sequence[Transformation],
        allowed_labels: list[int] | None = None,
    ) -> None:
        self.root: str = path
        self.input_transformations = input_transformations
        inputs, outputs, known_labels = self._get_data(input_name, output_name, allowed_labels)

        self.known_labels = known_labels
        self.inputs = torch.tensor(inputs)
        self.outputs = torch.tensor(outputs, dtype=torch.long)

    def _get_data(
        self,
        input_name: str | list[str],
        output_name: str | list[str],
        allowed_labels: list[int] | None,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        filenames: list[str] = sorted([name for name in os.listdir(self.root) if name.lower().endswith(".csv")])
        if not filenames:
            raise ValueError(f"Directory {self.root} must contain at least one CSV file")

        all_inputs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        for filename in filenames:
            df = pd.read_csv(os.path.join(self.root, filename))
            input_values = df[input_name].to_numpy()
            label_values = df[output_name].to_numpy().reshape(-1)

            for transformation in self.input_transformations:
                input_values = transformation(input_values)

            all_inputs.append(np.asarray(input_values))
            all_labels.append(np.asarray(label_values))

        inputs = np.concatenate(all_inputs)
        labels = np.concatenate(all_labels)
        if labels.ndim != 1:
            labels = labels.reshape(-1)

        if not np.all(np.equal(labels, labels.astype(int))):
            raise ValueError("signal classification labels must be integer class ids")
        label_ids = labels.astype(int)

        if allowed_labels is None:
            known_labels = sorted(np.unique(label_ids).tolist())
        else:
            known_labels = sorted(int(label) for label in allowed_labels)
            unknown_labels = sorted(set(label_ids.tolist()) - set(known_labels))
            if unknown_labels:
                raise ValueError(f"unknown label(s) found in {self.root}: {unknown_labels}")

        label_to_index = {label: index for index, label in enumerate(known_labels)}
        encoded_labels = np.array([label_to_index[label] for label in label_ids], dtype=np.int64)

        return inputs, encoded_labels, known_labels

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        input_tensor = self.inputs[index].view(-1)
        output_tensor = self.outputs[index]
        return input_tensor, output_tensor


EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


class Sample(NamedTuple):
    path: str
    label: str


class ClassificationSubset(Dataset):
    def __init__(
        self,
        path: str,
        input_transformations: Sequence[Transformation],
        grayscale: bool,
    ) -> None:
        self.path: str = path
        self.input_transformations = input_transformations
        self.grayscale: bool = grayscale

        self.classes: list[str] = scan_classes(path)
        self.samples: list[Sample] = []
        for class_name in self.classes:
            class_path: str = os.path.join(path, class_name)
            if not os.path.isdir(class_path):
                continue

            for root, _, files in os.walk(class_path):
                for file in files:
                    if not file.lower().endswith(EXTENSIONS):
                        continue

                    self.samples.append(Sample(os.path.join(root, file), class_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]

        with open(sample.path, "rb") as f:
            image = Image.open(f)
            image = image.convert("RGB" if not self.grayscale else "L")
        sample_array = np.array(image)

        for transformation in self.input_transformations:
            sample_array = transformation(sample_array)

        label: int = self.classes.index(sample.label)
        return torch.tensor(sample_array), label
