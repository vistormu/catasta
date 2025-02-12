from catasta import Scaffold, Dataset
from catasta.models import MambaImageClassifier
from catasta.dataclasses import EvalInfo


def main() -> None:
    model = MambaImageClassifier(
        input_shape=(28, 28, 3),
        n_classes=10,
        n_patches=7,
        d_model=16,
        d_state=16,
        d_conv=4,
        expand=4,
        n_layers=2,
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
