import os

from torch import Tensor

from catasta import Dataset, Scaffold
from catasta.dataclasses import EvalInfo, TrainInfo
from catasta.models import GPBinaryClassifier


def _epochs(default: int, smoke: int) -> int:
    return smoke if os.getenv("CATASTA_SMOKE", "0") == "1" else default


def predict_binary_labels(model: GPBinaryClassifier, x: Tensor) -> Tensor:
    latent_distribution = model(x)
    probability_distribution = model.likelihood(latent_distribution)
    return (probability_distribution.probs >= 0.5).long()


def main() -> None:
    model = GPBinaryClassifier(
        n_inducing_points=64,
        n_inputs=24,
        kernel="rq",
        mean="zero",
        likelihood="pg",
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
        loss_function="variational_elbo",
    )

    train_info: TrainInfo = scaffold.train(
        epochs=_epochs(120, 20),
        batch_size=64,
        lr=1e-2,
        early_stopping=("plateau", 15, 1e-4),
    )
    print(train_info)

    eval_info: EvalInfo = scaffold.evaluate(
        batch_size=64,
        prediction_function=predict_binary_labels,
    )
    print(eval_info)


if __name__ == "__main__":
    main()
