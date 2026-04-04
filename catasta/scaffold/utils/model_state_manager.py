from __future__ import annotations

from copy import deepcopy
from typing import Literal

from torch.nn import Module

EarlyStoppingMode = Literal["none", "trend", "plateau"]
EarlyStoppingPolicy = tuple[str, float | int, float | int]


class ModelStateManager:
    def __init__(self, mode: str | EarlyStoppingPolicy | None = "none") -> None:
        self.mode, self.alpha, self.patience, self.min_delta = self._parse_mode(mode)

        self.best_loss: float = float("inf")
        self.best_model_state: dict = {}

        self.stop: bool = False
        self.prev_loss: float | None = None
        self.derivative: float = 0.0
        self.bad_epochs: int = 0
        self.trend_positive_epochs: int = 0

    def _parse_mode(
        self,
        mode: str | EarlyStoppingPolicy | None,
    ) -> tuple[EarlyStoppingMode, float, int, float]:
        if mode is None or mode == "none":
            return "none", 0.0, 0, 0.0

        if not isinstance(mode, tuple) or len(mode) != 3 or not isinstance(mode[0], str):
            raise ValueError(
                "early_stopping must be one of: None, 'none', ('trend', alpha, patience), ('plateau', patience, min_delta)"
            )

        tag = mode[0].lower()
        if tag == "trend":
            alpha = float(mode[1])
            patience = int(mode[2])

            if not (0 <= alpha < 1):
                raise ValueError("trend alpha must be in [0, 1)")
            if patience < 1:
                raise ValueError("trend patience must be >= 1")

            return "trend", alpha, patience, 0.0

        if tag == "plateau":
            patience = int(mode[1])
            min_delta = float(mode[2])

            if patience < 1:
                raise ValueError("plateau patience must be >= 1")
            if min_delta < 0:
                raise ValueError("plateau min_delta must be >= 0")

            return "plateau", 0.0, patience, min_delta

        raise ValueError(
            "early_stopping must be one of: None, 'none', ('trend', alpha, patience), ('plateau', patience, min_delta)"
        )

    def __call__(self, model: Module, loss: float) -> None:
        previous_best_loss = self.best_loss

        if not self.best_model_state:
            self.best_model_state = deepcopy(model.state_dict())

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_model_state = deepcopy(model.state_dict())

        if self.mode == "none":
            return

        if self.mode == "trend":
            delta = 0.0 if self.prev_loss is None else loss - self.prev_loss
            self.derivative = self.alpha * self.derivative + (1 - self.alpha) * delta
            self.prev_loss = loss

            if self.derivative > 0:
                self.trend_positive_epochs += 1
            else:
                self.trend_positive_epochs = 0

            if self.trend_positive_epochs >= self.patience:
                self.stop = True
            return

        if self.mode == "plateau":
            if loss < (previous_best_loss - self.min_delta):
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1

            if self.bad_epochs >= self.patience:
                self.stop = True
            return

    def load_best_model_state(self, model: Module) -> None:
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)
