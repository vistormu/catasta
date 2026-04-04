import os

from catasta import Scaffold, Dataset
from catasta.models import FeedforwardRegressor
from catasta.dataclasses import TrainInfo


def _is_smoke() -> bool:
    return os.getenv("CATASTA_SMOKE", "0") == "1"


def _epochs(default: int, smoke: int) -> int:
    return smoke if _is_smoke() else default


def single_output() -> None:
    model = FeedforwardRegressor(
        n_inputs=1,
        n_outputs=1,
        hidden_dims=[16, 16, 16],
        dropout=0.1,
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
        loss_function="mse",
    )

    train_info: TrainInfo = scaffold.train(
        epochs=_epochs(1000, 5),
        batch_size=128,
        lr=1e-3,
    )
    print(train_info)

    info = scaffold.evaluate()
    print(info)


def multi_output() -> None:
    model = FeedforwardRegressor(
        n_inputs=4,
        n_outputs=3,
        hidden_dims=[16, 16, 16],
        dropout=0.1,
    )

    # dataset_root: str = "data/tactile/"
    dataset_root: str = "data/squishy/"
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
        loss_function="mse",
    )

    train_info: TrainInfo = scaffold.train(
        epochs=_epochs(5000, 5),
        batch_size=128,
        lr=1e-3,
    )
    print(train_info)

    info = scaffold.evaluate()
    print(info)


if __name__ == '__main__':
    single_output()
    if not _is_smoke():
        multi_output()
