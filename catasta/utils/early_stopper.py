from torch.nn import Module


class EarlyStopper:
    def __init__(self, patience: int, delta: float) -> None:
        self.patience: int = patience
        self.delta: float = delta
        self.best_loss: float = float("inf")
        self.counter: int = 0

    def __call__(self, loss: float) -> bool:
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True

        return False
