from typing import NamedTuple
import numpy as np


class RegressionTrainInfo(NamedTuple):
    train_losses: np.ndarray
    best_train_loss: float
    val_losses: np.ndarray | None
    best_val_loss: float | None
    lr_values: np.ndarray
