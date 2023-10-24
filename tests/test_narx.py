import os

import numpy as np
import pandas as pd

from torch import nn
from sysidentpy.neural_network import NARXNN
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation

from catasta.entities import ModelData


def get_data(dir: str) -> list[ModelData]:
    data: list[ModelData] = []
    for filename in os.listdir(dir):
        if not filename.endswith(".csv"):
            continue

        data_frame: pd.DataFrame = pd.read_csv(dir + filename)
        data.append(ModelData(
            input=data_frame['input'].to_numpy(),
            output=data_frame['output'].to_numpy(),
        ))

    return data


class NARX(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()

    def forward(self, xb):
        z = self.lin(xb)
        z = self.tanh(z)
        z = self.lin2(z)
        z = self.tanh(z)
        z = self.lin3(z)
        return z


basis_function = Polynomial(degree=1)

narx_net = NARXNN(
    net=NARX().to('cuda'),
    epochs=1000,
    learning_rate=0.01,
    ylag=2,
    xlag=2,
    basis_function=basis_function,
    model_type="NARMAX",
    loss_func='mse_loss',
    optimizer='Adam',
    verbose=True,
    optim_params={'betas': (0.9, 0.999), 'eps': 1e-05},
    device='cuda',
)


data: list[ModelData] = get_data("tests/data/nylon_strain/")
x_data = np.array([d.input for d in data])
y_data = np.array([d.output for d in data])

x_train = x_data[:-1].reshape(-1, 1)
y_train = y_data[:-1].reshape(-1, 1)

x_valid = x_data[-1:].reshape(-1, 1)
y_valid = y_data[-1:].reshape(-1, 1)

narx_net.fit(X=x_train, y=y_train, X_test=x_valid, y_test=y_valid)
yhat = narx_net.predict(X=x_valid, y=y_valid)

plot_results(y=y_valid, yhat=yhat, n=2500, style="seaborn-v0_8-darkgrid")
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$", style="seaborn-v0_8-darkgrid")
x1e = compute_cross_correlation(y_valid, yhat, x_valid)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$", style="seaborn-v0_8-darkgrid")
