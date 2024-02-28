from typing import NamedTuple
import numpy as np


class ClassificationTrainInfo(NamedTuple):
    train_losses: np.ndarray
    best_train_loss: float
    val_losses: np.ndarray
    best_val_loss: float
    train_accuracies: np.ndarray
    best_train_accuracy: float
    val_accuracies: np.ndarray
    best_val_accuracy: float
    lr_values: np.ndarray
