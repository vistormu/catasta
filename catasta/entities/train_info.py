from typing import NamedTuple


class TrainInfo(NamedTuple):
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
