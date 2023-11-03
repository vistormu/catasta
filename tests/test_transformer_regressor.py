import numpy as np

from catasta.models import TransformerRegressor
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.entities import RegressionEvalInfo, RegressionTrainInfo

from vclog import Logger


def main() -> None:
    n_dim: int = 32
    dataset = RegressionDataset(root="tests/data/steps/", n_dim=n_dim)
    model = TransformerRegressor(
        d_model=n_dim,
        n_heads=8,
        n_encoder_layers=6,
        n_decoder_layers=6,
        dim_feedforward=2048,
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
        train_split=6/7,
        lr=1e-3,
    )

    Logger.debug(f"min train loss: {np.min(train_info.train_loss):.4f}, "
                 f"min eval loss: {np.min(train_info.eval_loss):.4f}")  # type: ignore

    info: RegressionEvalInfo = scaffold.evaluate()

    Logger.debug(info)


if __name__ == '__main__':
    main()
