import os
import random
import shutil


EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ".csv")


def split_dataset(dataset: str, destination: str, task: str, splits: tuple[float, float, float], shuffle: bool) -> None:
    train_split: float = splits[0]
    val_split: float = splits[1]
    test_split: float = splits[2]

    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError("Splits must sum to 1.0")
    if not train_split:
        raise ValueError("Train split must be greater than 0")
    if not val_split and not test_split:
        raise ValueError("At least one of val split or test split must be greater than 0")

    if task == "classification":
        _classification_split(dataset, destination, shuffle, train_split, val_split, test_split)
    elif task == "regression":
        raise NotImplementedError("Regression task not implemented")
    else:
        raise ValueError(f"Task {task} not supported")


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
        os.makedirs(train_path, exist_ok=True)
        for img in train:
            train_img = os.path.join(train_path, os.path.basename(img))
            shutil.copyfile(img, train_img)

        val_path = os.path.join(destination, "validation", cls)
        os.makedirs(val_path, exist_ok=True) if val else None
        for img in val:
            val_img = os.path.join(val_path, os.path.basename(img))
            shutil.copyfile(img, val_img)

        test_path = os.path.join(destination, "testing", cls)
        os.makedirs(test_path, exist_ok=True) if test else None
        for img in test:
            test_img = os.path.join(test_path, os.path.basename(img))
            shutil.copyfile(img, test_img)
