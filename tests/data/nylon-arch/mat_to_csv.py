import os

import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DIR = "data/data_nylon/"


def kalman_filter(noisy_signal, A=1, H=1, Q=0.01, R=1, initial_state=None, initial_covariance=1):
    """
    Apply the Kalman filter to a noisy signal.

    Parameters:
    - noisy_signal: The observed noisy data points as a 1D numpy array.
    - A: The state transition model.
    - H: The observation model.
    - Q: The covariance of the process noise.
    - R: The covariance of the observation noise.
    - initial_state: Initial state estimate. Defaults to the first point of the noisy_signal.
    - initial_covariance: Initial error covariance.

    Returns:
    - filtered_signal: A 1D numpy array of filtered data points.
    """

    if initial_state is None:
        initial_state = noisy_signal[0]

    x = initial_state
    P = initial_covariance

    filtered_signal = []

    for measurement in noisy_signal:
        # Prediction
        x_pred = A * x
        P_pred = A * P * A + Q

        # Update
        K = P_pred * H / (H * P_pred * H + R)
        x = x_pred + K * (measurement - H * x_pred)
        P = (1 - K * H) * P_pred

        filtered_signal.append(x)

    return np.array(filtered_signal)


def main() -> None:
    for filename in os.listdir(DIR):
        if filename.endswith(".mat"):
            mat = sio.loadmat(DIR + filename)
        else:
            continue

        pos: np.ndarray = mat["posicion"][0][0][1][0][0][0].flatten()
        nylon: np.ndarray = mat["nylon_filtro"][0][0][1][0][0][0].flatten()
        ref: np.ndarray = mat["referencia"][0][0][1][0][0][0].flatten()

        # remove filter artifact
        nylon[:1000] = nylon[1000]

        # Normalize the data
        nylon = (nylon - np.min(nylon)) / (np.max(nylon) - np.min(nylon)) * 100
        pos = (pos - np.min(pos)) / (np.max(pos) - np.min(pos)) * 100
        ref = (ref - np.min(ref)) / (np.max(ref) - np.min(ref)) * 100

        # Kalman filter
        nylon = kalman_filter(nylon, Q=0.001, R=100)

        # Signal downsampling
        decimation_factor = 100
        nylon = nylon[::decimation_factor]
        pos = pos[::decimation_factor]
        ref = ref[::decimation_factor]

        # plt.plot(nylon)
        # plt.plot(pos)
        # plt.title(filename)
        # plt.show()

        data: dict[str, np.ndarray] = {
            "input": nylon,
            "output": pos,
        }

        pd.DataFrame(data).to_csv(DIR + filename.split(".")[0] + '.csv', index=False)


if __name__ == "__main__":
    main()
