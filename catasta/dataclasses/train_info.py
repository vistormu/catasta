from typing import NamedTuple
import numpy as np

from ..log import ansi


class TrainInfo(NamedTuple):
    """NamedTuple for storing training information.

    Attributes
    ----------
    train_losses : np.ndarray
        Array of the training losses for each epoch.
    best_train_loss : float
        The minimum value of the training losses.
    val_losses : np.ndarray
        Array of the validation losses for each epoch.
    best_val_loss : float
        The minimum value of the validation losses.
    lr_values : np.ndarray
        Array of the learning rates for each epoch.
    """
    train_losses: np.ndarray
    best_train_loss: float

    val_losses: np.ndarray
    best_val_loss: float

    lr_values: np.ndarray

    def __repr__(self) -> str:
        return f"\n{ansi.BOLD}{ansi.BLUE}-> training results{ansi.RESET}\n" \
            f"   |> best train loss: {self.best_train_loss:.4f}\n" \
            f"   |> best val loss:   {self.best_val_loss:.4f}"
