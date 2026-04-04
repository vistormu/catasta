from catasta import Dataset, Scaffold
from catasta.dataclasses import EvalInfo, TrainInfo
from catasta.models import ResNet1DRegressor


def main() -> None:
    model = ResNet1DRegressor(
        n_inputs=24,
        n_outputs=2,
        channels=[16, 32],
        blocks_per_stage=1,
        feedforward_dims=[32],
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
        epochs=80,
        batch_size=64,
        lr=1e-3,
        early_stopping=("plateau", 10, 1e-4),
    )
    print(train_info)

    eval_info: EvalInfo = scaffold.evaluate(batch_size=64)
    print(eval_info)


if __name__ == "__main__":
    main()
