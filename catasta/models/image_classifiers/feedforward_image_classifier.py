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

from einops import rearrange


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


class FeedforwardImageClassifier(Module):
    def __init__(self, *,
                 input_size: tuple[int, int, int],
                 dropout: float,
                 n_classes: int,
                 hidden_dims: list[int] = [],
                 use_layer_norm: bool = True,
                 use_batch_norm: bool = False,
                 activation: str = 'relu',
                 ) -> None:
        super().__init__()

        image_height, image_width, image_channels = input_size
        input_dim: int = image_height * image_width * image_channels

        layers: list[Module] = []

        # no hidden layers
        if not hidden_dims:
            self.net = Sequential(Linear(input_dim, n_classes))
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

        layers.append(Linear(hidden_dims[-1], n_classes))

        self.net: Sequential = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b ... -> b (...)")

        x = self.net(x)

        return x
