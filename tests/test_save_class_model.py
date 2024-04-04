from catasta.models import CNNClassifier
from catasta.datasets import ClassificationDataset
from catasta.scaffolds import ImageClassificationScaffold
from catasta.dataclasses import ClassificationEvalInfo

from vclog import Logger


def main() -> None:
    logger: Logger = Logger("catasta")

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

    dataset = ClassificationDataset(
        root="tests/data/mnist",
    )

    scaffold = ImageClassificationScaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="cross_entropy",
    )

    scaffold.train(
        epochs=10,
        batch_size=128,
        lr=1e-3,
        final_lr=None,
        early_stopping=None,
    )

    eval_info: ClassificationEvalInfo = scaffold.evaluate()
    logger.info(eval_info)

    scaffold.save(path="tests/models/", to_onnx=True)


if __name__ == '__main__':
    main()
