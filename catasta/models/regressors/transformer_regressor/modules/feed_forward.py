from torch import Tensor
from torch.nn import (
    Module,
    Linear,
    LayerNorm,
    Sequential,
    GELU,
    Dropout,
)


class FeedForward(Module):
    def __init__(self, *,
                 d_model: int,
                 hidden_dim: int,
                 dropout: float,
                 ) -> None:
        super().__init__()

        self.net = Sequential(
            LayerNorm(d_model),
            Linear(d_model, hidden_dim),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dim, d_model),
            Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
