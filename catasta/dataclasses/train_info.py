from typing import NamedTuple
import numpy as np


class TrainInfo(NamedTuple):
    """NamedTuple for storing training information.

    Attributes
    ----------
    train_losses : np.ndarray
        Array of the training losses for each epoch.
    best_train_loss : float
        The minimum value of the training losses.
    train_accuracies : np.ndarray
        Array of the training accuracies for each epoch. If the task is regression, this array will be empty.
    best_train_accuracy : float
        The maximum value of the training accuracies. If the task is regression, this value will be -inf.
    val_losses : np.ndarray
        Array of the validation losses for each epoch.
    best_val_loss : float
        The minimum value of the validation losses.
    val_accuracies : np.ndarray
        Array of the validation accuracies for each epoch. If the task is regression, this array will be empty.
    best_val_accuracy : float
        The maximum value of the validation accuracies. If the task is regression, this value will be -inf.
    lr_values : np.ndarray
        Array of the learning rates for each epoch.
    """
    train_losses: np.ndarray
    best_train_loss: float
    train_accuracies: np.ndarray
    best_train_accuracy: float

    val_losses: np.ndarray
    best_val_loss: float
    val_accuracies: np.ndarray
    best_val_accuracy: float

    lr_values: np.ndarray
