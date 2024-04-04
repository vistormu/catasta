import os
import numpy as np
from PIL import Image

from catasta.models import CNNClassifier
from catasta.archways import ClassificationArchway
from catasta.dataclasses import ClassificationEvalInfo


def vanilla() -> None:
    path: str = "tests/models/"

    model = CNNClassifier(
        input_shape=(28, 28, 3),
        n_classes=10,
        conv_out_channels=[32, 64],
        conv_kernel_sizes=[3, 3],
        conv_strides=[1, 1],
        conv_paddings=[1, 1],
        pooling_kernel_sizes=[2, 2],
        pooling_strides=[2, 2],
        pooling_paddings=[0, 0],
        feedforward_dims=[128, 64],
        dropout=0.5,
        activation="relu",
    )

    archway = ClassificationArchway(
        model=model,
        path=path,
    )

    imgs = []
    dataset_root = "tests/data/mnist/validation/0"
    for file in os.listdir(dataset_root):
        img = Image.open(os.path.join(dataset_root, file))
        img = np.array(img.convert("RGB"))
        imgs.append(img)

    input = np.array(imgs)
    prediction = archway.predict(input)

    predicted_output = prediction.argmax
    true_labels = np.zeros(len(predicted_output)).astype(int)

    info = ClassificationEvalInfo(
        true_labels=true_labels,
        predicted_labels=predicted_output,
        n_classes=10,
    )

    print(info)


def vanilla_onnx() -> None:
    path: str = "tests/models/"

    archway = ClassificationArchway(
        path=path,
        from_onnx=True,
    )

    imgs = []
    dataset_root = "tests/data/mnist/validation/0"
    for file in os.listdir(dataset_root):
        img = Image.open(os.path.join(dataset_root, file))
        img = np.array(img.convert("RGB"))
        imgs.append(img)

    input = np.array(imgs)
    prediction = archway.predict(input)

    predicted_output = prediction.argmax
    true_labels = np.zeros(len(predicted_output)).astype(int)

    info = ClassificationEvalInfo(
        true_labels=true_labels,
        predicted_labels=predicted_output,
        n_classes=10,
    )

    print(info)


if __name__ == "__main__":
    print("vanilla")
    vanilla()
    print("vanilla_onnx")
    vanilla_onnx()
