import numpy as np

from catasta.models import MambaImageClassifier
from catasta.datasets import ImageClassificationDataset
from catasta.scaffolds import ImageClassificationScaffold
from catasta.dataclasses import ClassificationTrainInfo, ClassificationEvalInfo
from catasta.transformations import Custom

from vclog import Logger


def main() -> None:
    logger: Logger = Logger("catasta")

    model = MambaImageClassifier(
        input_shape=(28, 28, 1),
        n_classes=10,
        n_patches=4,
        d_model=8,
        d_conv=4,
        d_state=8,
        n_layers=2,
        expand=2,
        layer_norm=False,
        use_fft=True,
    )

    def remove_channel(input: np.ndarray) -> np.ndarray:
        return input[..., :1]

    input_transformations = [
        Custom(remove_channel),
    ]

    dataset = ImageClassificationDataset(
        root="tests/data/mnist",
        input_transformations=input_transformations,  # type: ignore
    )

    scaffold = ImageClassificationScaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="cross_entropy",
        save_path=None,
    )

    train_info: ClassificationTrainInfo = scaffold.train(
        epochs=100,
        batch_size=128,
        lr=1e-4,
        final_lr=None,
        early_stopping=None,
    )

    logger.info(train_info.best_val_accuracy)

    eval_info: ClassificationEvalInfo = scaffold.evaluate()

    logger.info(eval_info)


if __name__ == '__main__':
    main()