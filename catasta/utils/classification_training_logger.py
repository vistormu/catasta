import numpy as np

from ..dataclasses import ClassificationTrainInfo


class ClassificationTrainingLogger:
    def __init__(self, epochs: int) -> None:
        self.epochs = epochs

        self.train_losses: list[float] = []
        self.train_accuracies: list[float] = []
        self.val_losses: list[float] = []
        self.val_accuracies: list[float] = []
        self.lr_values: list[float] = []

        self.avg_time_per_epoch: float = 0.0

    def log(self,
            train_loss: float,
            val_loss: float,
            train_accuracy: float,
            val_accuracy: float,
            lr: float,
            epoch: int,
            time_per_epoch: float
            ) -> None:
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)
        self.lr_values.append(lr)

        self.avg_time_per_epoch = (self.avg_time_per_epoch * (epoch - 1) + time_per_epoch) / epoch

    def get_info(self) -> ClassificationTrainInfo:
        return ClassificationTrainInfo(
            train_losses=np.array(self.train_losses),
            best_train_loss=min(self.train_losses),
            val_losses=np.array(self.val_losses),
            best_val_loss=min(self.val_losses),
            train_accuracies=np.array(self.train_accuracies),
            best_train_accuracy=max(self.train_accuracies),
            val_accuracies=np.array(self.val_accuracies),
            best_val_accuracy=max(self.val_accuracies),
            lr_values=np.array(self.lr_values),
        )

    def __repr__(self) -> str:
        epoch: str = f"{len(self.train_losses)}/{self.epochs}"
        percentage: float = len(self.train_losses) / self.epochs
        percentage_str: str = f"{int(percentage * 100)}%" if percentage > 0.09 else f"0{int(percentage*100)}%"

        train_loss: str = f"{self.train_losses[-1]:.4f}"
        val_loss: str = f"{self.val_losses[-1]:.4f}"
        best_val_loss: str = f"{min(self.val_losses):.4f}"
        train_accuracy: str = f"{self.train_accuracies[-1]:.4f}"
        val_accuracy: str = f"{self.val_accuracies[-1]:.4f}"
        best_val_accuracy: str = f"{max(self.val_accuracies):.4f}"

        lr: str = f"{self.lr_values[-1]:.4f}"

        time_remaining: int = int(self.avg_time_per_epoch * (self.epochs - len(self.train_losses)))
        time_remaining_str: str = f"{int(time_remaining/60)}m {time_remaining%60}s"

        return f"epoch {epoch} ({percentage_str}) | loss[train, val, best][{train_loss}, {val_loss}, {best_val_loss}] | acc[train, val, best][{train_accuracy}, {val_accuracy}, {best_val_accuracy}] | lr: {lr} | {time_remaining_str}"
