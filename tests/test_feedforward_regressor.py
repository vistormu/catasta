import numpy as np
import matplotlib.pyplot as plt

from catasta.models import FeedforwardRegressor
from catasta.datasets import ModelDataset
from catasta.scaffolds import RegressorScaffold
from catasta.entities import EvalInfo


def main() -> None:
    n_dim: int = 20
    dataset: ModelDataset = ModelDataset(root="tests/data/nylon_strain/", n_dim=n_dim)

    model: FeedforwardRegressor = FeedforwardRegressor(input_dim=n_dim,
                                                       hidden_dims=[64, 32, 16],
                                                       output_dim=1,
                                                       dropout=0.1,
                                                       )

    scaffold: RegressorScaffold = RegressorScaffold(model, dataset)

    losses: np.ndarray = scaffold.train(epochs=100,
                                        batch_size=256,
                                        train_split=6/7,
                                        lr=1e-3,
                                        )

    plt.figure(figsize=(30, 20))
    plt.plot(losses)
    plt.show()

    info: EvalInfo = scaffold.evaluate()

    print(info)


if __name__ == "__main__":
    main()
