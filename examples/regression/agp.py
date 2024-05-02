import pandas as pd
import matplotlib.pyplot as plt

from catasta import Scaffold, CatastaDataset, Archway
from catasta.models import ApproximateGPRegressor
from catasta.scaffolds import Scaffold
from catasta.dataclasses import PredictionInfo, TrainInfo


def main() -> None:
    # Model
    model = ApproximateGPRegressor(
        n_inducing_points=128,
        context_length=1,
        kernel="matern",
        mean="constant",
    )

    # Dataset
    dataset_root: str = "data/incomplete/"
    dataset = CatastaDataset(dataset_root, task="regression")

    # Training
    scaffold = Scaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="variational_elbo",
    )

    train_info: TrainInfo = scaffold.train(
        epochs=5000,
        batch_size=128,
        lr=1e-3,
        early_stopping=True,
    )
    print(f"min train loss: {train_info.best_train_loss:.4f}")

    info = scaffold.evaluate()
    print(info)

    # save model
    save_model_path: str = "saved_models/"
    scaffold.save(save_model_path)

    # Inference
    df = pd.read_csv(dataset_root + "data.csv")
    observed_input = df["input"].to_numpy()
    observed_output = df["output"].to_numpy()
    true_input = df["true_input"].to_numpy()
    true_output = df["true_output"].to_numpy()

    archway = Archway(
        path=save_model_path+model.__class__.__name__,
        verbose=False,
    )
    predictions: PredictionInfo = archway.predict(true_input.reshape(-1, 1))

    predicted_output = predictions.value
    predicted_std = predictions.std

    # Plot
    plt.plot(true_input, true_output, color="red", label="true")
    plt.plot(observed_input, observed_output, "k.", label="observations")
    plt.plot(true_input, predicted_output, color="blue", label="predicted")
    plt.fill_between(
        true_input,
        predicted_output - predicted_std,
        predicted_output + predicted_std,
        color="blue",
        alpha=0.2,
    )
    plt.show()


if __name__ == '__main__':
    main()
