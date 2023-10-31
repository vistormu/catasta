from catasta.models import TransformerImageClassifier
from catasta.scaffolds import ClassifierScaffold
from catasta.datasets import ImageDataset
from catasta.entities import TrainInfo
from catasta.datasets.classifier_dataset import ClassifierDataset

import matplotlib.pyplot as plt


def main() -> None:
    # Model parameters
    sequence_length: int = 256
    n_heads: int = 4
    n_embeddings: int = 256
    n_layers: int = 6
    feedforward_dim: int = 512

    # Training parameters
    batch_size: int = 8
    learning_rate: float = 6e-4
    epochs: int = 20
    log_interval: int = 1
    eval_iters: int = 10

    # dataset = ImageDataset(root="tests/data/images_resized/")
    dataset = ClassifierDataset(root="tests/data/images_resized/")

    model = TransformerImageClassifier(embedding_dim=n_embeddings,
                                       patch_size=sequence_length//32,
                                       n_heads=n_heads,
                                       n_layers=n_layers,
                                       image_size=sequence_length,
                                       feedforward_dim=feedforward_dim,
                                       n_classes=dataset.n_classes,
                                       )

    scaffold = ClassifierScaffold(model=model, dataset=dataset)

    # Train model
    train_info_list: list[TrainInfo] = scaffold.train(batch_size=batch_size,
                                                      learning_rate=learning_rate,
                                                      epochs=epochs,
                                                      log_interval=log_interval,
                                                      eval_iters=eval_iters,
                                                      )


if __name__ == "__main__":
    main()
