from torch.nn import Module


class ModelStateManager:
    def __init__(self, patience_and_delta: tuple[int, float] | None) -> None:
        self.patience: int | None = patience_and_delta[0] if patience_and_delta is not None else None
        self.delta: float | None = patience_and_delta[1] if patience_and_delta is not None else None

        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.best_model_states: list[dict] = []
        self.early_stop: bool = False

    def __call__(self, models: list[Module], loss: float):
        if not self.best_model_states:
            self.best_model_states = [model.state_dict() for model in models]

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_model_states = [model.state_dict() for model in models]
            self.counter = 0

        elif self.delta is not None and self.patience is not None:
            if loss - self.best_loss > self.delta:
                self.counter += 1

        if self.patience is not None and self.delta is not None:
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model_state(self, models: list[Module]) -> None:
        for i, model in enumerate(models):
            model.load_state_dict(self.best_model_states[i])
