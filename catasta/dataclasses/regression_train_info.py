from typing import NamedTuple
import numpy as np


class RegressionTrainInfo(NamedTuple):
    train_loss: np.ndarray
    eval_loss: np.ndarray | None = None
