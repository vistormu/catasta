from catasta import Scaffold, CatastaDataset
from catasta.models import ApproximateGPRegressor
from catasta.dataclasses import TrainInfo


def main() -> None:
    model = ApproximateGPRegressor(
        n_inducing_points=128,
        context_length=1,
        kernel="matern",
        mean="constant",
    )

    dataset_root: str = "data/incomplete/"
    dataset = CatastaDataset(dataset_root, task="regression")

    scaffold = Scaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="variational_elbo",
    )

    train_info: TrainInfo = scaffold.train(
        epochs=100,
        batch_size=128,
        lr=1e-3,
    )
    print(train_info)

    info = scaffold.evaluate()
    print(info)


if __name__ == '__main__':
    main()
