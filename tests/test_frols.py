import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)

from vclog import Logger


def main() -> None:
    # df1 = pd.read_csv("tests/data/sysidentpy/x_cc.csv")
    # df2 = pd.read_csv("tests/data/sysidentpy/y_cc.csv")

    # x_train, x_valid = np.split(df1.iloc[::25].values, 2)
    # y_train, y_valid = np.split(df2.iloc[::25].values, 2)

    data_dir = "tests/data/nylon_strain/"
    # data_dir = "tests/data/steps/"
    x_data = np.array([])
    y_data = np.array([])
    for filename in os.listdir(data_dir):
        if not filename.endswith(".csv"):
            continue

        data_frame: pd.DataFrame = pd.read_csv(data_dir + filename)
        x_data = np.concatenate((x_data, data_frame['input'].to_numpy().flatten()))
        y_data = np.concatenate((y_data, data_frame['output'].to_numpy().flatten()))

    x_data = x_data.reshape(-1, 1)
    y_data = y_data.reshape(-1, 1)

    # x_data = x_data[::5].reshape(-1, 1)
    # y_data = y_data[::5].reshape(-1, 1)

    # x_data, y_data = y_data, x_data

    assert len(x_data) == len(y_data)

    # index: int = int(len(x_data)*0.99)
    # x_train, x_valid = x_data[:index], x_data[index:]
    # y_train, y_valid = y_data[:index], y_data[index:]

    x_train, x_valid = np.split(x_data, 2)
    y_train, y_valid = np.split(y_data, 2)

    assert len(x_train) == len(y_train)
    assert len(x_valid) == len(y_valid)

    Logger.debug(f"x_train.shape: {x_train.shape}")
    Logger.debug(f"x_valid.shape: {x_valid.shape}")
    Logger.debug(f"y_train.shape: {y_train.shape}")
    Logger.debug(f"y_valid.shape: {y_valid.shape}")

    basis_function = Polynomial(degree=3)

    model = FROLS(
        order_selection=True,
        n_info_values=20,
        extended_least_squares=False,
        ylag=5,
        xlag=5,
        info_criteria="bic",
        estimator="recursive_least_squares",
        basis_function=basis_function,
    )

    model.fit(X=x_train, y=y_train)
    yhat = model.predict(X=x_valid, y=y_valid)
    rrse = root_relative_squared_error(y_valid, yhat)
    print(rrse)

    r = pd.DataFrame(
        results(
            model.final_model,
            model.theta,
            model.err,
            model.n_terms,
            err_precision=8,
            dtype="sci",
        ),
        columns=["Regressors", "Parameters", "ERR"],
    )
    print(r)

    plot_results(y=y_valid, yhat=yhat, n=1000, style="seaborn-v0_8-darkgrid")
    # ee = compute_residues_autocorrelation(y_valid, yhat)
    # plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$", style="seaborn-v0_8-darkgrid")
    # x1e = compute_cross_correlation(y_valid, yhat, x_valid)
    # plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$", style="seaborn-v0_8-darkgrid")


if __name__ == "__main__":
    main()
