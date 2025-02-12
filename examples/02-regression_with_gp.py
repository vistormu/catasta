from catasta import Scaffold, Dataset
from catasta.models import GPRegressor
from catasta.dataclasses import TrainInfo


def single_output() -> None:
    model = GPRegressor(
        n_inducing_points=128,
        n_inputs=1,
        n_outputs=1,
        kernel="matern",
        mean="constant",
    )

    dataset_root: str = "data/incomplete/"
    dataset = Dataset(
        dataset_root,
        task="regression",
        input_name="input",
        output_name="output",
    )

    scaffold = Scaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="variational_elbo",
    )

    train_info: TrainInfo = scaffold.train(
        epochs=1000,
        batch_size=128,
        lr=1e-3,
    )
    print(train_info)

    info = scaffold.evaluate()
    print(info)


def multi_output() -> None:
    model = GPRegressor(
        n_inducing_points=128,
        n_inputs=4,
        n_outputs=3,
        kernel="matern",
        mean="constant",
    )

    dataset_root: str = "data/tactile/"
    dataset = Dataset(
        dataset_root,
        task="regression",
        input_name=["s0", "s1", "s2", "s3"],
        output_name=["x", "y", "fz"],
    )

    scaffold = Scaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="variational_elbo",
    )

    train_info: TrainInfo = scaffold.train(
        epochs=1000,
        batch_size=128,
        lr=1e-3,
    )
    print(train_info)

    info = scaffold.evaluate()
    print(info)


if __name__ == '__main__':
    single_output()
    multi_output()
