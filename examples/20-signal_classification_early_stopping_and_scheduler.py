import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from catasta import Scaffold, Dataset
from catasta.models import FeedforwardRegressor
from catasta.dataclasses import EvalInfo, TrainInfo


def train_with_plateau_policy() -> None:
    model = FeedforwardRegressor(
        n_inputs=24,
        n_outputs=2,
        hidden_dims=[32, 32],
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

    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    scaffold = Scaffold(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        loss_function="cross_entropy",
    )

    train_info: TrainInfo = scaffold.train(
        epochs=150,
        batch_size=64,
        lr=1e-3,
        scheduler=(scheduler, "val_loss"),
        early_stopping=("plateau", 12, 1e-4),
    )
    print(train_info)

    eval_info: EvalInfo = scaffold.evaluate(batch_size=32)
    print(eval_info)


def train_with_trend_policy() -> None:
    model = FeedforwardRegressor(
        n_inputs=24,
        n_outputs=2,
        hidden_dims=[32, 32],
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
        epochs=150,
        batch_size=64,
        lr=1e-3,
        early_stopping=("trend", 0.95, 3),
    )
    print(train_info)

    eval_info: EvalInfo = scaffold.evaluate(batch_size=32)
    print(eval_info)


if __name__ == '__main__':
    train_with_plateau_policy()
    train_with_trend_policy()
