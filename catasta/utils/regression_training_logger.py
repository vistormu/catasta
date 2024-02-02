import numpy as np

from ..dataclasses import RegressionTrainInfo


class RegressionTrainingLogger:
    def __init__(self, epochs: int) -> None:
        self.epochs: int = epochs

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.lr_values: list[float] = []

        self.avg_time_per_epoch: float = 0.0

    def log(self,
            train_loss: float,
            val_loss: float | None,
            lr: float,
            epoch: int,
            time_per_epoch: float,
            ) -> None:
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        self.lr_values.append(lr)

        self.avg_time_per_epoch = (self.avg_time_per_epoch * (epoch - 1) + time_per_epoch) / epoch

    def get_regression_train_info(self) -> RegressionTrainInfo:
        return RegressionTrainInfo(
            train_losses=np.array(self.train_losses),
            best_train_loss=np.min(self.train_losses),
            val_losses=np.array(self.val_losses) if len(self.val_losses) > 0 else None,
            best_val_loss=np.min(self.val_losses) if len(self.val_losses) > 0 else None,
            lr_values=np.array(self.lr_values),
        )

    def __repr__(self) -> str:
        epoch: str = f"{len(self.train_losses)}/{self.epochs}"
        percentage: float = len(self.train_losses) / self.epochs
        percentage_str: str = f"{int(percentage * 100)}%" if percentage > 0.09 else f"0{int(percentage * 100)}%"

        train_loss: str = f"{self.train_losses[-1]:.4f}"
        val_loss: str = f"{self.val_losses[-1]:.4f}" if len(self.val_losses) > 0 else "-"
        best_val_loss: str = f"{min(self.val_losses):.4f}" if len(self.val_losses) > 0 else "-"
        lr_str: str = f"{self.lr_values[-1]:.2e}"

        time_remaining: int = int(self.avg_time_per_epoch * (self.epochs - len(self.train_losses)))
        time_remaining_str: str = f"{int(time_remaining/60)}m {time_remaining%60}s"

        return f"epoch {epoch} ({percentage_str}) | loss[train, val, best] [{train_loss}, {val_loss}, {best_val_loss}] | lr: {lr_str} | {time_remaining_str}"
