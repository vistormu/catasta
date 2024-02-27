import numpy as np

from catasta.models import FeedforwardImageClassifier
from catasta.datasets import ImageClassificationDataset
from catasta.scaffolds.classification_scaffold.vanilla_classification_scaffold import VanillaClassificationScaffold
from catasta.dataclasses import ClassificationTrainInfo, ClassificationEvalInfo
from catasta.transformations import Custom

from vclog import Logger


def main() -> None:
    logger: Logger = Logger("catasta")

    model = FeedforwardImageClassifier(
        input_dim=28*28,
        n_classes=10,
        hidden_dims=[16, 16, 16],
        activation="relu",
        dropout=0.1,
    )

    def remove_channel(input: np.ndarray) -> np.ndarray:
        return np.mean(input, axis=2)

    input_transformations = [
        Custom(remove_channel),
    ]

    dataset = ImageClassificationDataset(
        root="tests/data/mnist",
        input_transformations=input_transformations,  # type: ignore
    )

    scaffold = VanillaClassificationScaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="cross_entropy",
        save_path=None,
    )

    train_info: ClassificationTrainInfo = scaffold.train(
        epochs=10,
        batch_size=128,
        lr=1e-3,
        final_lr=None,
        early_stopping=None,
    )

    logger.info(train_info.best_val_accuracy)

    eval_info: ClassificationEvalInfo = scaffold.evaluate()

    logger.info(eval_info)


if __name__ == '__main__':
    main()
