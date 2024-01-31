import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from catasta.models import TransformerRegressor, ApproximateGPRegressor, FeedforwardRegressor
from catasta.archways import RegressionArchway
from catasta.transformations import (
    WindowSliding,
    Normalization,
    Decimation,
    Custom,
)


def main() -> None:
    path: str = "tests/models/"

    n_dim: int = 16
    # model = TransformerRegressor(
    #     context_length=n_dim,
    #     n_patches=2,
    #     d_model=64,
    #     n_heads=2,
    #     n_layers=4,
    #     feedforward_dim=32,
    #     head_dim=2,
    #     dropout=0.0,
    #     use_fft=True,
    # )
    model = FeedforwardRegressor(
        input_dim=n_dim,
        hidden_dims=[8, 16, 8],
        dropout=0.0,
    )
    # model = ApproximateGPRegressor(
    #     n_inducing_points=16,
    #     n_inputs=n_dim,
    #     kernel="rq",
    #     mean="constant"
    # )

    archway = RegressionArchway(
        model=model,
        path=path,
    )

    df = pd.read_csv("tests/data/nylon_elastic/strain/tri_30.csv")
    input = df["input"].to_numpy().flatten()

    transformations = [
        Custom(lambda x: x[500_000:1_500_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=100),
        WindowSliding(window_size=n_dim, stride=1),
    ]

    for t in transformations:
        input = t(input)

    predicted = []
    for i in input:
        prediction = archway.predict(i)
        predicted.append(prediction.prediction)

    prediction = np.array(predicted).flatten()

    plt.plot(prediction)
    plt.show()


if __name__ == "__main__":
    main()
