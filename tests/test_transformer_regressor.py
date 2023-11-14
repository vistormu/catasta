import numpy as np
import matplotlib.pyplot as plt

from catasta.models import TransformerRegressor
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.entities import RegressionEvalInfo, RegressionTrainInfo

from vclog import Logger


def main() -> None:
    n_dim: int = 16
    dataset = RegressionDataset(
        root="tests/data/steps/",
        context_length=n_dim,
        prediction_length=n_dim,
        splits=(0.8, 0.1, 0.1),
    )
    model = TransformerRegressor(
        input_dim=n_dim,
        output_dim=n_dim,
        d_model=64,
        n_heads=2,
        n_encoder_layers=2,
        n_decoder_layers=2,
        dim_feedforward=128,
    )
    scaffold = RegressionScaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="huber",
    )

    train_info: RegressionTrainInfo = scaffold.train(
        epochs=100,
        batch_size=256,
        lr=1e-3,
    )

    Logger.debug(f"min train loss: {np.min(train_info.train_loss):.4f}, "
                 f"min eval loss: {np.min(train_info.eval_loss):.4f}")  # type: ignore

    info: RegressionEvalInfo = scaffold.evaluate()

    plt.plot(info.predicted, label="predictions")
    plt.plot(info.real, label="real")
    plt.show()

    Logger.debug(info)


if __name__ == '__main__':
    main()
