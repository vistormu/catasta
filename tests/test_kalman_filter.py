import numpy as np
import matplotlib.pyplot as plt

from catasta.transformations import KalmanFilter


def main() -> None:
    noisy_sin = np.sin(np.linspace(0, 100, 1000)) + np.random.normal(0, 0.1, 1000)

    kf = KalmanFilter(
        F=np.array([[1, 1], [0, 1]]),
        H=np.array([[1, 0]]),
        Q=np.array([[0.0001, 0], [0, 0.0001]]),
        R=np.array([[1]]),
    )

    predictions = kf(noisy_sin)

    plt.figure(figsize=(30, 20))
    plt.plot(noisy_sin, label="noisy sin", color="red")
    plt.plot(predictions, label="kalman filter", color="blue")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
