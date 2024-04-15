import numpy as np
import pandas as pd


noise_mean: float = 0.0
noise_std: float = 0.02
n_samples: int = 1000

true_input: np.ndarray = np.linspace(-0.5, 1.5, n_samples)
input = np.hstack([np.linspace(-0.2, 0.2, n_samples//2), np.linspace(0.6, 1, n_samples//2)])


def f(x: np.ndarray, noise: np.ndarray) -> np.ndarray:
    return x + 0.3 * np.sin(2 * np.pi * (x + noise)) + 0.2 * np.sin(4 * np.pi * (x + noise)) + noise


output = f(input, np.random.normal(noise_mean, noise_std, n_samples))
true_output = f(true_input, np.zeros_like(true_input))

data = {
    "input": input,
    "output": output,
    "true_input": true_input,
    "true_output": true_output,
}

pd.DataFrame(data).to_csv("data.csv", index=False)
