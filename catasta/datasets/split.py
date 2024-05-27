import os
import random
import shutil

import pandas as pd


EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ".csv")


def split_dataset(dataset: str,
                  destination: str,
                  task: str,
                  splits: tuple[float, float, float],
                  shuffle: bool,
                  ) -> None:
    """ Split a dataset into training, validation, and testing sets.

    Parameters
    ----------
    dataset : str
        The path to the dataset.
    destination : str
        The path to save the split dataset.
    task : str
        The task type, either "classification" or "regression".
    splits : tuple[float, float, float]
        The split ratios for training, validation, and testing sets.
    shuffle : bool
        Whether to shuffle the dataset before splitting.

    Raises
    ------
    ValueError
        If the splits do not sum to 1.0, if the training split is 0, or if both the validation and testing splits are 0.
        If the task is not supported.
        If the task is classification and the dataset is not a directory.
        If the task is regression and the dataset is not a file.
    """
    train_split: float = splits[0]
    val_split: float = splits[1]
    test_split: float = splits[2]

    _check_arguments(dataset, task, train_split, val_split, test_split)

    os.makedirs(os.path.join(destination, "training"), exist_ok=True)
    os.makedirs(os.path.join(destination, "validation"), exist_ok=True) if val_split else None
    os.makedirs(os.path.join(destination, "testing"), exist_ok=True) if test_split else None

    split_function = _classification_split if task == "classification" else _regression_split
    split_function(dataset, destination, shuffle, train_split, val_split, test_split)


def _check_arguments(dataset: str,
                     task: str,
                     train_split: float,
                     val_split: float,
                     test_split: float,
                     ) -> None:
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError("Splits must sum to 1.0")
    if not train_split:
        raise ValueError("Train split must be greater than 0")
    if not val_split and not test_split:
        raise ValueError("At least one of val split or test split must be greater than 0")
    if task not in ("classification", "regression"):
        raise ValueError(f"Task {task} not supported")
    if task == "classification" and not os.path.isdir(dataset):
        raise ValueError("Dataset must be a directory")
    if task == "regression" and not os.path.isfile(dataset):
        raise ValueError("Dataset must be a file")


def _regression_split(dataset: str,
                      destination: str,
                      shuffle: bool,
                      train_split: float,
                      val_split: float,
                      test_split: float,
                      ) -> None:
    df = pd.read_csv(dataset)
    n_samples = len(df)
    indices = list(range(n_samples))

    if shuffle:
        random.shuffle(indices)

    train_index = int(train_split * n_samples)
    val_index = train_index + int(val_split * n_samples)
    test_index = val_index + int(test_split * n_samples)

    train = df.iloc[indices[:train_index]]
    val = df.iloc[indices[train_index:val_index]] if val_split else pd.DataFrame()
    test = df.iloc[indices[val_index:test_index]] if test_split else pd.DataFrame()

    filename = os.path.basename(dataset)

    train_path = os.path.join(destination, "training", filename)
    train.to_csv(train_path, index=False)

    val_path = os.path.join(destination, "validation", filename)
    val.to_csv(val_path, index=False) if val_split else None

    test_path = os.path.join(destination, "testing", filename)
    test.to_csv(test_path, index=False) if test_split else None


def _classification_split(dataset: str,
                          destination: str,
                          shuffle: bool,
                          train_split: float,
                          val_split: float,
                          test_split: float,
                          ) -> None:
    classes = os.listdir(dataset)
    classes = [cls for cls in classes if os.path.isdir(os.path.join(dataset, cls))]

    for cls in classes:
        cls_path = os.path.join(dataset, cls)
        images = [os.path.join(cls_path, img) for img in os.listdir(cls_path)]
        images = [img for img in images if img.lower().endswith(EXTENSIONS)]

        if shuffle:
            random.shuffle(images)

        n_images = len(images)
        train_index = int(train_split * n_images)
        val_index = train_index + int(val_split * n_images)
        test_index = val_index + int(test_split * n_images)

        train = images[:train_index]
        val = images[train_index:val_index] if val_split else []
        test = images[val_index:test_index] if test_split else []

        train_path = os.path.join(destination, "training", cls)
        for img in train:
            train_img = os.path.join(train_path, os.path.basename(img))
            shutil.copyfile(img, train_img)

        val_path = os.path.join(destination, "validation", cls)
        for img in val:
            val_img = os.path.join(val_path, os.path.basename(img))
            shutil.copyfile(img, val_img)

        test_path = os.path.join(destination, "testing", cls)
        for img in test:
            test_img = os.path.join(test_path, os.path.basename(img))
            shutil.copyfile(img, test_img)
