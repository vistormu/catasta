import numpy as np

from catasta.models import FeedforwardRegressor
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.entities import RegressionEvalInfo, RegressionTrainInfo

from vclog import Logger


def main() -> None:
    n_dim: int = 20
    dataset = RegressionDataset(
        root="tests/data/steps/",
        context_length=n_dim,
        prediction_length=1,
        splits=(0.8, 0.1, 0.1),
    )
    model = FeedforwardRegressor(
        input_dim=n_dim,
        hidden_dims=[64, 32, 16],
        output_dim=1,
        dropout=0.1,
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

    Logger.debug(info)


if __name__ == '__main__':
    main()
