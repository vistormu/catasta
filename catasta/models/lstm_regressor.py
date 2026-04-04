from torch import Tensor
from torch.nn import (
    Dropout,
    GELU,
    LSTM,
    Linear,
    Module,
    ModuleList,
    ReLU,
    Sigmoid,
    Tanh,
)


def get_activation_function(activation: str) -> Module:
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
            raise ValueError(f'activation function {activation} not supported')


class LSTMRegressor(Module):
    def __init__(self, *,
                 n_inputs: int,
                 n_outputs: int,
                 hidden_dim: int,
                 n_layers: int = 1,
                 bidirectional: bool = False,
                 feedforward_dims: list[int] = [],
                 dropout: float = 0.0,
                 activation: str = 'relu',
                 ) -> None:
        super().__init__()

        if n_layers <= 0:
            raise ValueError('n_layers must be greater than 0')

        self.n_outputs = n_outputs
        self.lstm = LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        recurrent_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.dense_layers = ModuleList()
        for i, ff_dim in enumerate(feedforward_dims):
            linear = Linear(
                in_features=recurrent_output_dim if i == 0 else feedforward_dims[i - 1],
                out_features=ff_dim,
            )
            activation_function = get_activation_function(activation)
            dropout_layer = Dropout(p=dropout)
            self.dense_layers.append(linear)
            self.dense_layers.append(activation_function)
            self.dense_layers.append(dropout_layer)

        final_in_features = recurrent_output_dim if len(feedforward_dims) == 0 else feedforward_dims[-1]
        self.output_layer = Linear(final_in_features, n_outputs)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        elif x.ndim != 3:
            raise ValueError('input must have shape [batch, sequence] or [batch, sequence, features]')

        x, _ = self.lstm(x)
        x = x[:, -1, :]

        for layer in self.dense_layers:
            x = layer(x)

        x = self.output_layer(x)
        return x.squeeze(-1) if self.n_outputs == 1 else x
