from torch.nn import Module


class ModelStateManager:
    def __init__(self, early_stopping: bool) -> None:
        self.early_stopping = early_stopping

        self.best_loss: float = float("inf")
        self.best_model_states: list[dict] = []

        self.alpha: float = 0.95
        self.stop: bool = False

        self.prev_loss: float = float("inf")
        self.derivative: float = 0.0

    def __call__(self, models: list[Module], loss: float):
        if not self.best_model_states:
            self.best_model_states = [model.state_dict() for model in models]

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_model_states = [model.state_dict() for model in models]

        unfiltered_derivate: float = loss - self.prev_loss if self.prev_loss != float("inf") else 0.0
        self.derivative = self.alpha * self.derivative + (1 - self.alpha) * unfiltered_derivate

        self.prev_loss = loss

        if self.early_stopping:
            if self.derivative > 0:
                self.stop = True

    def load_best_model_state(self, models: list[Module]) -> None:
        for i, model in enumerate(models):
            model.load_state_dict(self.best_model_states[i])
