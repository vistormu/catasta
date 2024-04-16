import numpy as np

from ...dataclasses import TrainInfo


class TrainingLogger:
    def __init__(self, epochs: int) -> None:
        self.epochs: int = epochs

        self.train_losses: list[float] = []
        self.train_accuracies: list[float] = []

        self.val_losses: list[float] = []
        self.val_accuracies: list[float] = []

        self.lr_values: list[float] = []

        self.avg_time_per_epoch: float = 0.0

    def log(self,
            train_loss: float,
            train_accuracy: float | None,
            val_loss: float | None,
            val_accuracy: float | None,
            lr: float,
            epoch: int,
            time_per_epoch: float,
            ) -> None:
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_accuracy) if train_accuracy is not None else None

        self.val_losses.append(val_loss) if val_loss is not None else None
        self.val_accuracies.append(val_accuracy) if val_accuracy is not None else None

        self.lr_values.append(lr)

        self.avg_time_per_epoch = (self.avg_time_per_epoch * (epoch - 1) + time_per_epoch) / epoch

    def get_info(self) -> TrainInfo:
        return TrainInfo(
            train_losses=np.array(self.train_losses),
            train_accuracies=np.array(self.train_accuracies),
            best_train_loss=np.min(self.train_losses),
            best_train_accuracy=np.max(self.train_accuracies) if len(self.train_accuracies) > 0 else 0.0,
            val_losses=np.array(self.val_losses),
            val_accuracies=np.array(self.val_accuracies),
            best_val_loss=np.min(self.val_losses) if len(self.val_losses) > 0 else np.inf,
            best_val_accuracy=np.max(self.val_accuracies) if len(self.val_accuracies) > 0 else 0.0,
            lr_values=np.array(self.lr_values),
        )

    def __repr__(self) -> str:
        epoch: str = f"{len(self.train_losses)}/{self.epochs}"
        percentage: float = len(self.train_losses) / self.epochs
        percentage_str: str = f"{int(percentage * 100)}%" if percentage > 0.09 else f"0{int(percentage * 100)}%"
        epoch_msg: str = f"epoch {epoch} ({percentage_str}) |"

        # LOSS
        train_loss: str = f"{self.train_losses[-1]:.4f}"
        val_loss: str = f"{self.val_losses[-1]:.4f}" if len(self.val_losses) > 0 else "-"
        best_val_loss: str = f"{min(self.val_losses):.4f}" if len(self.val_losses) > 0 else "-"
        loss_msg: str = f"loss[train, val, best] [{train_loss}, {val_loss}, {best_val_loss}] |"

        # ACCURACY
        train_accuracy: str = f"{self.train_accuracies[-1]:.4f}" if len(self.train_accuracies) > 0 else ""
        val_accuracy: str = f"{self.val_accuracies[-1]:.4f}" if len(self.val_accuracies) > 0 else ""
        best_val_accuracy: str = f"{max(self.val_accuracies):.4f}" if len(self.val_accuracies) > 0 else ""
        accuracy_msg: str = f"accuracy[train, val, best] [{train_accuracy}, {val_accuracy}, {best_val_accuracy}] |" if len(self.train_accuracies) > 0 else ""

        # LR
        lr_str: str = f"{self.lr_values[-1]:.2e}"
        lr_msg: str = f"lr: {lr_str} |"

        # TIME
        time_remaining: int = int(self.avg_time_per_epoch * (self.epochs - len(self.train_losses)))
        time_remaining_str: str = f"{int(time_remaining/60)}m {time_remaining%60}s"
        time_msg: str = f"{time_remaining_str}"

        return f"{epoch_msg} {loss_msg} {accuracy_msg} {lr_msg} {time_msg}"
