from catasta import Scaffold, Dataset
from catasta.models import FeedforwardImageClassifier
from catasta.dataclasses import EvalInfo


def main() -> None:
    model = FeedforwardImageClassifier(
        input_shape=(28, 28, 3),
        n_classes=10,
        hidden_dims=[16, 16, 16],
        activation="relu",
        dropout=0.1,
    )

    dataset = Dataset(
        "data/reduced_mnist",
        task="classification",
    )

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
    )

    eval_info: EvalInfo = scaffold.evaluate(batch_size=32)
    print(eval_info)


if __name__ == '__main__':
    main()
