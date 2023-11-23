class ModelStateManager:
    def __init__(self, *,
                 patience: int,
                 delta: float,
                 ) -> None:
        self.patience: int = patience
        self.delta: float = delta
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.best_model_state: dict = {}
        self.early_stop: bool = False

    def __call__(self, model_state: dict, loss: float):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            self.best_model_state = model_state
        elif loss - self.best_loss > self.delta:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

    def stop(self) -> bool:
        return self.early_stop

    def get_best_model_state(self) -> dict:
        return self.best_model_state
