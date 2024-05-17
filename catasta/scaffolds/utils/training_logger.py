import numpy as np

from ...dataclasses import TrainInfo


class TrainingLogger:
    def __init__(self, task: str, epochs: int) -> None:
        self.task: str = task
        self.epochs: int = epochs

        self.train_losses: list[float] = []
        self.train_accuracies: list[float] = []

        self.val_losses: list[float] = []
        self.val_accuracies: list[float] = []

        self.lr_values: list[float] = []

        self.avg_time_per_epoch: float = 0.0

    def log(self,
            train_loss: float,
            train_accuracy: float,
            val_loss: float,
            val_accuracy: float,
            lr: float,
            epoch: int,
            time_per_epoch: float,
            ) -> None:
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_accuracy) if self.task == "classification" else None

        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy) if self.task == "classification" else None

        self.lr_values.append(lr)

        self.avg_time_per_epoch = (self.avg_time_per_epoch * (epoch - 1) + time_per_epoch) / epoch

    def get_info(self) -> TrainInfo:
        return TrainInfo(
            train_losses=np.array(self.train_losses),
            train_accuracies=np.array(self.train_accuracies) if self.task == "classification" else np.array([]),
            best_train_loss=np.min(self.train_losses),
            best_train_accuracy=np.max(self.train_accuracies) if self.task == "classification" else -np.inf,
            val_losses=np.array(self.val_losses),
            val_accuracies=np.array(self.val_accuracies) if self.task == "classification" else np.array([]),
            best_val_loss=np.min(self.val_losses),
            best_val_accuracy=np.max(self.val_accuracies) if self.task == "classification" else -np.inf,
            lr_values=np.array(self.lr_values),
        )

    def __repr__(self) -> str:
        epoch: int = len(self.train_losses)
        percentage: float = epoch / self.epochs
        percentage_str: str = f"{int(percentage * 100)}%" if percentage > 0.09 else f"0{int(percentage * 100)}%"
        epoch_msg: str = f"    -> epoch:    {epoch}/{self.epochs} ({percentage_str})\n"

        # LOSS
        train_loss: str = f"{self.train_losses[-1]:.4f}"
        val_loss: str = f"{self.val_losses[-1]:.4f}"
        best_val_loss: str = f"{min(self.val_losses):.4f}"
        loss_msg: str = f"    -> loss:     train({train_loss}), val({val_loss}), best({best_val_loss})\n"

        # ACCURACY
        train_accuracy: str = f"{self.train_accuracies[-1]:.4f}" if len(self.train_accuracies) > 0 else ""
        val_accuracy: str = f"{self.val_accuracies[-1]:.4f}" if len(self.val_accuracies) > 0 else ""
        best_val_accuracy: str = f"{max(self.val_accuracies):.4f}" if len(self.val_accuracies) > 0 else ""
        accuracy_msg: str = f"    -> accuracy: train({train_accuracy}), val({val_accuracy}), best({best_val_accuracy})\n" if len(self.train_accuracies) > 0 else ""

        # TIME
        time_remaining: int = int(self.avg_time_per_epoch * (self.epochs - len(self.train_losses)))
        time_remaining_str: str = f"{int(time_remaining/60)}m {time_remaining%60}s"
        time_msg: str = f"    -> time:     {time_remaining_str}"

        clear_line: str = "\x1b[2K" + "\r" + "\x1b[1A"

        msg: str = ""
        if epoch != 1:
            msg += clear_line*4 if self.task == "classification" else clear_line*3

        msg += epoch_msg + loss_msg + accuracy_msg + time_msg

        return msg
