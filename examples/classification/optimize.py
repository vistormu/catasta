import numpy as np

from catasta import CatastaDataset, Scaffold, Foundation
from catasta.models import TransformerClassifier
from catasta.transformations import Custom


def objective(params: dict) -> float:
    model = TransformerClassifier(
        input_shape=(28, 28, 1),
        n_classes=10,
        n_patches=params["n_patches"],
        d_model=params["d_model"],
        n_layers=params["n_layers"],
        n_heads=params["n_heads"],
        feedforward_dim=params["feedforward_dim"],
        head_dim=params["head_dim"],
        dropout=params["dropout"],
        layer_norm=params["layer_norm"],
    )

    input_transformations = [
        Custom(lambda x: np.expand_dims(x, axis=-1)),
        Custom(lambda x: x / 255.0),
    ]
    dataset = CatastaDataset(
        root="data/reduced_mnist",
        task="classification",
        input_transformations=input_transformations,
        grayscale=True,
    )

    scaffold = Scaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="cross_entropy",
        verbose=False,
    )

    scaffold.train(
        epochs=100,
        batch_size=128,
        lr=1e-3,
        early_stopping_alpha=0.95,
    )

    eval_info = scaffold.evaluate()

    return eval_info.accuracy


def main() -> None:
    hp_space = {
        "n_patches": (2, 7),
        "d_model": (8, 16),
        "n_layers": (1, 2),
        "n_heads": (1, 2),
        "feedforward_dim": (8, 16),
        "head_dim": (4, 8),
        "dropout": (0.0, 0.5),
        "layer_norm": (True, False),
    }

    foundation = Foundation(
        hyperparameter_space=hp_space,
        objective_function=objective,
        sampler="bogp",
        n_trials=100,
        direction="maximize",
        use_secretary=True,
        catch_exceptions=True,
    )

    optimization_info = foundation.optimize()

    print(optimization_info.best_hyperparameters)


if __name__ == '__main__':
    main()
