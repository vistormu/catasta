from catasta import Foundation, Dataset, Scaffold
from catasta.models import GPRegressor


def objective(params: dict, objective_extra_parameters: dict) -> float:
    dataset_root: str = objective_extra_parameters["dataset_root"]

    model = GPRegressor(
        n_inducing_points=params["n_inducing_points"],
        n_inputs=1,
        n_outputs=1,
        kernel=params["kernel"],
        mean=params["mean"],
    )

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
        verbose=False,
    )

    scaffold.train(
        epochs=100,
        batch_size=32,
        lr=1e-3,
    )

    info = scaffold.evaluate()

    return info.rmse


def main() -> None:
    hp_space = {
        "n_inducing_points": (16, 32),
        "kernel": ("matern", "rbf"),
        "mean": ("constant", "zero"),
    }

    foundation = Foundation(
        hyperparameter_space=hp_space,
        objective_function=objective,
        objective_extra_parameters={"dataset_root": "data/incomplete/"},
        sampler="bogp",
        n_trials=10,
        direction="minimize",
        catch_exceptions=True,
    )

    optimization_info = foundation.optimize()

    print(optimization_info)


if __name__ == "__main__":
    main()
