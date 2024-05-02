import os

import numpy as np
from PIL import Image

from catasta import Scaffold, CatastaDataset, Archway
from catasta.models import CNNClassifier
from catasta.dataclasses import EvalInfo


def main() -> None:
    model = CNNClassifier(
        input_shape=(28, 28, 3),
        n_classes=10,
        conv_out_channels=[32, 64],
        conv_kernel_size=3,
        conv_stride=1,
        conv_padding=1,
        pooling_kernel_size=2,
        pooling_stride=2,
        pooling_padding=0,
        feedforward_dims=[128, 64],
        dropout=0.5,
        activation="relu",
    )

    dataset = CatastaDataset("data/reduced_mnist", task="classification")

    scaffold = Scaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="cross_entropy",
    )

    scaffold.train(
        epochs=100,
        batch_size=128,
        lr=1e-3,
        early_stopping=True,
    )

    eval_info = scaffold.evaluate()
    print(eval_info)

    save_model_path = "saved_models/"
    scaffold.save(save_model_path)

    archway = Archway(
        path=save_model_path+model.__class__.__name__,
    )

    imgs = []
    dataset_root = "data/reduced_mnist/val/0"
    for file in os.listdir(dataset_root):
        img = Image.open(os.path.join(dataset_root, file))
        img = np.array(img.convert("RGB"))
        imgs.append(img)

    input = np.array(imgs)
    prediction = archway.predict(input)

    predicted_output = prediction.argmax
    true_labels = np.zeros(len(predicted_output)).astype(int)

    info = EvalInfo(
        task="classification",
        true_output=true_labels,
        predicted_output=predicted_output,
        n_classes=10,
    )

    print(info)
    print(info.confusion_matrix)


if __name__ == '__main__':
    main()
