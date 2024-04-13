import pandas as pd

from catasta.models import TransformerRegressor, ApproximateGPRegressor
from catasta.archways import RegressionArchway
from catasta.transformations import (
    WindowSliding,
    Normalization,
    Decimation,
    Custom,
    Slicing,
)
from catasta.dataclasses import RegressionEvalInfo


def vanilla() -> None:
    path: str = "tests/models/"

    n_dim: int = 768
    model = TransformerRegressor(
        context_length=n_dim,
        n_patches=8,
        d_model=8,
        n_heads=4,
        n_layers=2,
        feedforward_dim=4,
        head_dim=4,
        dropout=0.0,
    )

    archway = RegressionArchway(
        model=model,
        path=path,
    )

    df = pd.read_csv("tests/data/nylon_elastic/strain/tri_30.csv")
    input = df["input"].to_numpy().flatten()
    output = df["output"].to_numpy().flatten()

    input_transformations = [
        Custom(lambda x: x[1_500_000: 2_000_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=100),
        WindowSliding(window_size=n_dim, stride=1),
    ]
    output_transformations = [
        Custom(lambda x: x[1_500_000: 2_000_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=100),
        Slicing(amount=n_dim - 1, end="left"),
    ]

    for t in input_transformations:
        input = t(input)
    for t in output_transformations:
        output = t(output)

    prediction = archway.predict(input)

    predicted_output = prediction.value
    true_output = output[-len(predicted_output):]
    true_input = input[-len(predicted_output):]

    info = RegressionEvalInfo(
        true_input=true_input,
        predicted_output=predicted_output,
        true_output=true_output,
    )

    print(info)


def vanilla_onnx() -> None:
    path: str = "tests/models/"

    n_dim: int = 768
    archway = RegressionArchway(
        path=path,
        from_onnx=True,
    )

    df = pd.read_csv("tests/data/nylon_elastic/strain/tri_30.csv")
    input = df["input"].to_numpy().flatten()
    output = df["output"].to_numpy().flatten()

    input_transformations = [
        Custom(lambda x: x[1_500_000: 2_000_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=100),
        WindowSliding(window_size=n_dim, stride=1),
    ]
    output_transformations = [
        Custom(lambda x: x[1_500_000: 2_000_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=100),
        Slicing(amount=n_dim - 1, end="left"),
    ]

    for t in input_transformations:
        input = t(input)
    for t in output_transformations:
        output = t(output)

    prediction = archway.predict(input)

    predicted_output = prediction.value
    true_output = output[-len(predicted_output):]
    true_input = input[-len(predicted_output):]

    info = RegressionEvalInfo(
        true_input=true_input,
        predicted_output=predicted_output,
        true_output=true_output,
    )

    print(info)


def gp() -> None:
    path: str = "tests/models/gp/"

    n_dim: int = 768
    model = ApproximateGPRegressor(
        context_length=n_dim,
        n_inducing_points=128,
    )

    archway = RegressionArchway(
        model=model,
        path=path,
    )

    df = pd.read_csv("tests/data/nylon_elastic/strain/tri_30.csv")
    input = df["input"].to_numpy().flatten()
    output = df["output"].to_numpy().flatten()

    input_transformations = [
        Custom(lambda x: x[1_500_000: 2_000_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=100),
        WindowSliding(window_size=n_dim, stride=1),
    ]
    output_transformations = [
        Custom(lambda x: x[1_500_000: 2_000_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=100),
        Slicing(amount=n_dim - 1, end="left"),
    ]

    for t in input_transformations:
        input = t(input)
    for t in output_transformations:
        output = t(output)

    prediction = archway.predict(input)

    predicted_output = prediction.value
    true_output = output[-len(predicted_output):]
    true_input = input[-len(predicted_output):]

    info = RegressionEvalInfo(
        true_input=true_input,
        predicted_output=predicted_output,
        true_output=true_output,
    )

    print(info)


if __name__ == "__main__":
    # print("vanilla")
    # vanilla()
    # print("vanilla_onnx")
    # vanilla_onnx()
    # print("gp")
    gp()
