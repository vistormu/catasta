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
    BatchNorm1d,
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
                 use_layer_norm: bool = True,
                 use_batch_norm: bool = False,
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
            # linear
            layers.append(Linear(hidden_dims[i-1], hidden_dims[i]))

            # batch norm
            if use_batch_norm:
                layers.append(BatchNorm1d(hidden_dims[i]))

            # layer norm
            if use_layer_norm:
                layers.append(LayerNorm(hidden_dims[i]))

            # activation
            layers.append(get_actvation_function(activation))

            # dropout
            layers.append(Dropout(dropout))

        layers.append(Linear(hidden_dims[-1], 1))

        self.net: Sequential = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)

        return x.squeeze()
