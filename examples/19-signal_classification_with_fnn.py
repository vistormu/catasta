import os

from catasta import Scaffold, Dataset
from catasta.models import FeedforwardRegressor
from catasta.dataclasses import EvalInfo, TrainInfo


def _epochs(default: int, smoke: int) -> int:
    return smoke if os.getenv("CATASTA_SMOKE", "0") == "1" else default


def main() -> None:
    model = FeedforwardRegressor(
        n_inputs=24,
        n_outputs=2,
        hidden_dims=[16, 16],
        dropout=0.1,
    )

    dataset_root: str = "data/italy_power_demand_signal_classification"
    input_columns: list[str] = [f"s{i}" for i in range(24)]
    dataset = Dataset(
        dataset_root,
        task="signal_classification",
        input_name=input_columns,
        output_name="label",
    )

    scaffold = Scaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="cross_entropy",
    )

    train_info: TrainInfo = scaffold.train(
        epochs=_epochs(100, 10),
        batch_size=128,
        lr=1e-3,
        early_stopping=("plateau", 10, 1e-4),
    )
    print(train_info)

    eval_info: EvalInfo = scaffold.evaluate(batch_size=32)
    print(eval_info)


if __name__ == '__main__':
    main()
