from catasta import Scaffold, CatastaDataset
from catasta.models import CNNClassifier
from catasta.dataclasses import TrainInfo


def main() -> None:
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

    dataset = CatastaDataset("tests/data/mnist", task="classification")

    scaffold = Scaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="cross_entropy",
    )

    train_info: TrainInfo = scaffold.train(
        epochs=10,
        batch_size=128,
        lr=1e-3,
    )

    print(train_info.best_val_accuracy)

    eval_info = scaffold.evaluate()

    print(eval_info)


if __name__ == '__main__':
    main()
