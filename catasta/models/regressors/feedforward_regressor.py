from torch import Tensor
from torch.nn import (
    Module,
    Linear,
    Sequential,
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    LayerNorm,
    Dropout,
)


def get_actvation_function(activation: str) -> Module:
    match activation:
        case 'relu':
            return ReLU()
        case 'sigmoid':
            return Sigmoid()
        case 'tanh':
            return Tanh()
        case 'gelu':
            return GELU()
        case _:
            raise ValueError(f'Activation function {activation} not supported')


class FeedforwardRegressor(Module):
    def __init__(self, *,
                 input_dim: int,
                 dropout: float,
                 hidden_dims: list[int] = [],
                 use_norm: bool = True,
                 activation: str = 'relu',
                 ) -> None:
        super().__init__()

        layers: list[Module] = []

        # no hidden layers
        if not hidden_dims:
            self.net = Sequential(Linear(input_dim, 1))
            return

        # hidden layers
        layers.append(Linear(input_dim, hidden_dims[0]))
        n_layers = len(hidden_dims)

        for i in range(1, n_layers):
            layers.append(Linear(hidden_dims[i-1], hidden_dims[i]))
            if use_norm:
                layers.append(LayerNorm(hidden_dims[i]))
            layers.append(get_actvation_function(activation))
            layers.append(Dropout(dropout))

        layers.append(Linear(hidden_dims[-1], 1))

        self.net: Sequential = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)

        return x.squeeze()
