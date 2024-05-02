import os

import numpy as np
from PIL import Image

from catasta import Scaffold, CatastaDataset, Archway
from catasta.models import TransformerClassifier
from catasta.dataclasses import EvalInfo


def main() -> None:
    model = TransformerClassifier(
        input_shape=(28, 28, 3),
        n_classes=10,
        n_patches=4,
        d_model=16,
        n_layers=2,
        n_heads=2,
        feedforward_dim=16,
        head_dim=4,
        dropout=0.5,
        layer_norm=True,
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


if __name__ == '__main__':
    main()
