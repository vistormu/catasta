import os

import torch
from torch.nn import Module

from vclog import Logger


class ModelStateManager:
    def __init__(self, patience_and_delta: tuple[int, float] | None, save_path: str | None) -> None:
        self.patience: int | None = patience_and_delta[0] if patience_and_delta is not None else None
        self.delta: float | None = patience_and_delta[1] if patience_and_delta is not None else None

        self.save_path: str | None = save_path
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.best_model_state: dict = {}
        self.early_stop: bool = False

        self.logger: Logger = Logger("catasta")

    def __call__(self, model_state: dict, loss: float):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            self.best_model_state = model_state
        elif self.delta is not None and self.patience is not None:
            if loss - self.best_loss > self.delta:
                self.counter += 1

        if self.patience is not None and self.delta is not None:
            if self.counter >= self.patience:
                self.early_stop = True

        if not self.best_model_state:
            self.best_model_state = model_state

    def stop(self) -> bool:
        return self.early_stop

    def save_models(self, models: list[Module]) -> None:
        if self.save_path is None:
            return

        for model in models:
            model_name: str = model.__class__.__name__
            model_path: str = os.path.join(self.save_path, f"{model_name}.pt")
            torch.save(model.state_dict(), model_path)

            self.logger.info(f"saved model {model_name} to {model_path}")

    def load_best_model_state(self, model: Module) -> None:
        model.load_state_dict(self.best_model_state)
