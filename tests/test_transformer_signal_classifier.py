from catasta.models import TransformerSignalClassifier
from catasta.scaffolds import ClassifierScaffold
from catasta.datasets import SignalDataset
from catasta.entities import TrainInfo

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

    dataset = SignalDataset(root="tests/signals", sequence_length=sequence_length)

    model = TransformerSignalClassifier(embedding_dim=n_embeddings,
                                        patch_size=sequence_length//32,
                                        n_heads=n_heads,
                                        n_layers=n_layers,
                                        sequence_length=sequence_length,
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

    # Plot training info
    train_loss_list: list[float] = [train_info.train_loss for train_info in train_info_list]
    val_loss_list: list[float] = [train_info.val_loss for train_info in train_info_list]
    train_accuracy_list: list[float] = [train_info.train_acc for train_info in train_info_list]
    val_accuracy_list: list[float] = [train_info.val_acc for train_info in train_info_list]

    plt.figure()
    plt.plot(train_loss_list, label='Train loss')
    plt.plot(val_loss_list, label='Validation loss')
    plt.legend()

    plt.figure()
    plt.plot(train_accuracy_list, label='Train accuracy')
    plt.plot(val_accuracy_list, label='Validation accuracy')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
