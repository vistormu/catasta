import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pysindy as ps

# data: pd.DataFrame = pd.read_csv("tests/data/nylon_carmen/strain/ms_16000us_10y20mhz_silicona.csv")
data: pd.DataFrame = pd.read_csv("tests/data/steps/step_1.csv")
x: np.ndarray = data["input"].to_numpy().flatten()
y: np.ndarray = data["output"].to_numpy().flatten()

model: ps.SINDy = ps.SINDy(optimizer=ps.STLSQ(threshold=0.1, alpha=0.5, fit_intercept=True))

model.fit(y, t=0.0094, u=x, unbias=False, quiet=False)

model.print()

y_pred: np.ndarray = model.predict(y, u=x).flatten()

# y_pred = y_pred[0] - y_pred

r2: float = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2).astype(float)
print(f"R2: {r2}")

plt.plot(y, label="train")
plt.plot(y_pred, label="pred")
plt.legend()
plt.show()
