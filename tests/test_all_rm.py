import os

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostRegressor
from sklearn.linear_model import BayesianRidge, ARDRegression
from sysidentpy.general_estimators import NARX

from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.model_structure_selection import FROLS

from vclog import Logger

basis_function = Polynomial(degree=2)


estimators = [
    (
        "KNeighborsRegressor",
        NARX(
            base_estimator=KNeighborsRegressor(),
            xlag=10,
            ylag=10,
            basis_function=basis_function,
            model_type="NARMAX",
        ),
    ),
    (
        "NARX-DecisionTreeRegressor",
        NARX(
            base_estimator=DecisionTreeRegressor(),
            xlag=10,
            ylag=10,
            basis_function=basis_function,
        ),
    ),
    (
        "NARX-RandomForestRegressor",
        NARX(
            base_estimator=RandomForestRegressor(n_estimators=200),
            xlag=10,
            ylag=10,
            basis_function=basis_function,
        ),
    ),
    (
        "NARX-Catboost",
        NARX(
            base_estimator=CatBoostRegressor(
                iterations=800, learning_rate=0.1, depth=8
            ),
            xlag=10,
            ylag=10,
            basis_function=basis_function,
            fit_params={"verbose": False},
        ),
    ),
    # (
    #     "NARX-ARD",
    #     NARX(
    #         base_estimator=ARDRegression(),
    #         xlag=10,
    #         ylag=10,
    #         basis_function=basis_function,
    #     ),
    # ),
    (
        "FROLS-Polynomial_NARX",
        FROLS(
            order_selection=True,
            n_info_values=50,
            extended_least_squares=False,
            ylag=10,
            xlag=10,
            info_criteria="bic",
            estimator="recursive_least_squares",
            basis_function=basis_function,
        ),
    ),
]

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

resultados = {}
for nome_do_modelo, modelo in estimators:
    resultados["%s" % (nome_do_modelo)] = []
    modelo.fit(X=x_train, y=y_train)
    yhat = modelo.predict(X=x_valid, y=y_valid)
    result = root_relative_squared_error(
        y_valid[modelo.max_lag:], yhat[modelo.max_lag:]
    )
    resultados["%s" % (nome_do_modelo)].append(result)
    print(nome_do_modelo, "%.3f" % np.mean(result))
