import numpy as np
import time

from ...dataclasses import TrainInfo
from ...log import ansi


def format_time(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")

    return " ".join(parts)


class TrainingLogger:
    def __init__(self, task: str, epochs: int) -> None:
        self.task: str = task
        self.epochs: int = epochs

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

        self.lr_values: list[float] = []

        self.avg_time_per_epoch: float = 0.0
        self.start_time: float = time.time()

    def log(self,
            train_loss: float,
            val_loss: float,
            lr: float,
            epoch: int,
            time_per_epoch: float,
            ) -> None:
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.lr_values.append(lr)

        self.avg_time_per_epoch = (self.avg_time_per_epoch * (epoch - 1) + time_per_epoch) / epoch

    def get_info(self) -> TrainInfo:
        return TrainInfo(
            train_losses=np.array(self.train_losses),
            best_train_loss=np.min(self.train_losses),
            val_losses=np.array(self.val_losses),
            best_val_loss=np.min(self.val_losses),
            lr_values=np.array(self.lr_values),
        )

    def __repr__(self) -> str:
        # progress
        epoch: int = len(self.train_losses)
        percentage: int = int(epoch / self.epochs * 100)

        # loss
        train_loss: str = f"{self.train_losses[-1]:.4f}"
        val_loss: str = f"{self.val_losses[-1]:.4f}"
        best_val_loss: str = f"{min(self.val_losses):.4f}"

        # time
        time_from_start: int = int(time.time() - self.start_time)
        time_remaining: int = int(self.avg_time_per_epoch * (self.epochs - len(self.train_losses)))

        # message
        clear = ansi.START + (ansi.UP + ansi.CLEAR_LINE)*3
        epoch_msg = f"   |> epoch:   {epoch}/{self.epochs} ({percentage}%)\n"
        loss_msg = f"   |> loss:    train({train_loss}), val({val_loss}), best({best_val_loss})\n"
        time_msg = f"   |> time:    {format_time(time_from_start)} ({format_time(time_remaining)} remaining)"

        return f"{clear if epoch != 1 else ''}{epoch_msg}{loss_msg}{time_msg}"
