from catasta import Foundation, Dataset, Scaffold
from catasta.models import GPRegressor


def objective(params: dict, extra: dict) -> float:
    # check that the extra parameters are being passed correctly
    if extra["name"] != "gp":
        return -10_000

    model = GPRegressor(
        n_inducing_points=params["n_inducing_points"],
        n_inputs=1,
        n_outputs=1,
        kernel=params["kernel"],
        mean=params["mean"],
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
        verbose=False,
    )

    scaffold.train(
        epochs=100,
        batch_size=32,
        lr=1e-3,
    )

    info = scaffold.evaluate(prediction_function=lambda model, x: model(x).mean.cpu().numpy())

    return info.rmse[0]


def main() -> None:
    hp_space = {
        "n_inducing_points": (16, 32),
        "kernel": ("matern", "rbf"),
        "mean": ("constant", "zero"),
    }

    extra = {
        "name": "gp",
    }

    foundation = Foundation(
        hyperparameter_space=hp_space,
        objective_function=objective,
        objective_extra_parameters=extra,
        sampler="bogp",
        n_trials=10,
        direction="minimize",
        catch_exceptions=True,
    )

    optimization_info = foundation.optimize()

    print(optimization_info)


if __name__ == "__main__":
    main()
