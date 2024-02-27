import os
from typing import NamedTuple

import numpy as np
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset

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
        if "val" in split:
            validation_dir_name = split
        if "test" in split:
            test_dir_name = split

    return train_dir_name, validation_dir_name, test_dir_name


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


class Sample(NamedTuple):
    path: str
    label: str


class ImageClassificationSubset(Dataset):
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
                    if not file.endswith(IMG_EXTENSIONS):
                        continue

                    sample = Sample(os.path.join(root, file), class_name)
                    self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample: Sample = self.samples[index]

        with open(sample.path, "rb") as f:
            img: Image.Image = Image.open(f)
            img = img.convert("RGB")

        sample_array: np.ndarray = np.array(img)
        for transformation in self.input_transformations:
            sample_array = transformation(sample_array)

        label: int = self.classes.index(sample.label)

        return torch.tensor(sample_array), label


class ImageClassificationDataset:
    def __init__(self, *,
                 root: str,
                 input_transformations: list[Transformation] = [],
                 ) -> None:
        self.root: str = root if root.endswith("/") else root + "/"

        splits: tuple[str, str, str] = scan_splits(self.root)
        self.train_split: str = splits[0]
        self.validation_split: str = splits[1]
        self.test_split: str = splits[2]

        self.train: Dataset = ImageClassificationSubset(self.root + self.train_split,
                                                        input_transformations,
                                                        )
        self.validation: Dataset | None = ImageClassificationSubset(self.root + self.validation_split,
                                                                    input_transformations,
                                                                    ) if self.validation_split else None
        self.test: Dataset | None = ImageClassificationSubset(self.root + self.test_split,
                                                              input_transformations,
                                                              ) if self.test_split else None

        self._check_arguments()

    def _check_arguments(self) -> None:
        if not os.path.isdir(self.root):
            raise ValueError(f"root must be a directory. Found {self.root}")
        if not self.train_split:
            raise ValueError(f"a directory named 'train' must be present in {self.root}")
        if not self.validation_split and not self.test_split:
            raise ValueError(f"at least a validation or test split must be present in {self.root}")

        # same number of classes in all splits
        if len(self.train.classes) == 0:  # type: ignore
            raise ValueError(f"no classes found in {self.train.path}")  # type: ignore

        if self.validation:
            if len(self.train.classes) != len(self.validation.classes):  # type: ignore
                raise ValueError(f"number of classes in train and validation splits must be the same")

        if self.test:
            if len(self.train.classes) != len(self.test.classes):  # type: ignore
                raise ValueError(f"number of classes in train and test splits must be the same")
