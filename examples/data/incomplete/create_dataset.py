import os

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

# divide into train and val
n_train = 800
n_val = n_samples - n_train

# take random samples
train_indices = np.random.choice(n_samples, n_train, replace=False)
val_indices = np.setdiff1d(np.arange(n_samples), train_indices)

train_true_input = true_input[train_indices]
train_true_output = true_output[train_indices]
train_input = input[train_indices]
train_output = output[train_indices]

val_true_input = true_input[val_indices]
val_true_output = true_output[val_indices]
val_input = input[val_indices]
val_output = output[val_indices]

data = {
    "input": input,
    "output": output,
    "true_input": true_input,
    "true_output": true_output,
}

pd.DataFrame(data).to_csv("data.csv", index=False)

train_data = {
    "input": train_input,
    "output": train_output,
    "true_input": train_true_input,
    "true_output": train_true_output,
}

if not os.path.exists("train"):
    os.makedirs("train")

pd.DataFrame(train_data).to_csv("train/data.csv", index=False)

val_data = {
    "input": val_input,
    "output": val_output,
    "true_input": val_true_input,
    "true_output": val_true_output,
}

if not os.path.exists("val"):
    os.makedirs("val")

pd.DataFrame(val_data).to_csv("val/data.csv", index=False)
