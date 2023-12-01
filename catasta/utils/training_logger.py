from vclog import Logger


def log_train_data(train_loss: float,
                   val_loss: float,
                   best_val_loss: float,
                   lr: float,
                   epoch: int,
                   epochs: int,
                   percentage: int,
                   time_per_batch: float,
                   time_per_epoch: float,
                   ) -> None:
    train_loss_str: str = f" {train_loss:.4f}" if train_loss > 0 else f"{train_loss:.4f}"
    val_loss_str: str = f"{val_loss:.4f}" if val_loss > 0 else f"{val_loss:.4f}"
    best_val_loss_str: str = f"{best_val_loss:.4f}" if best_val_loss > 0 else f"{best_val_loss:.4f}"
    lr_str: str = f"{lr:.4f}"
    epoch_str: str = f"{epoch}/{epochs}"
    time_str: str = f"{time_per_batch:.4f}"
    percentage_str: str = f"{percentage}%" if percentage > 9.0 else f"0{percentage}%"
    time_remaining_str: str = f"{int(time_per_epoch * (epochs - epoch)/1000)}"

    log_str: str = f"epoch {epoch_str} ({percentage_str})| loss[train, val, best] [{train_loss_str}, {val_loss_str}, {best_val_loss_str}] | lr: {lr_str} | {time_str} ms/batch | {time_remaining_str}s remaining"

    Logger("catasta").info(log_str, flush=True)
