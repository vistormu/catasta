import torch
from torch import Tensor
from torch.nn import (
    Conv1d,
    Dropout,
    Flatten,
    GELU,
    Linear,
    MaxPool1d,
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


class CNNRegressor(Module):
    def __init__(self, *,
                 n_inputs: int,
                 n_outputs: int,
                 conv_out_channels: list[int],
                 feedforward_dims: list[int] = [],
                 conv_kernel_size: int = 3,
                 conv_stride: int = 1,
                 conv_padding: int = 1,
                 pooling_kernel_size: int = 2,
                 pooling_stride: int = 2,
                 pooling_padding: int = 0,
                 dropout: float = 0.0,
                 activation: str = 'relu',
                 ) -> None:
        super().__init__()
        self.n_outputs = n_outputs

        self.conv_layers = ModuleList()
        for i, out_channels in enumerate(conv_out_channels):
            conv1d = Conv1d(
                in_channels=1 if i == 0 else conv_out_channels[i - 1],
                out_channels=out_channels,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                padding=conv_padding,
            )
            activation_function = get_activation_function(activation)
            max_pool = MaxPool1d(
                kernel_size=pooling_kernel_size,
                stride=pooling_stride,
                padding=pooling_padding,
            )
            self.conv_layers.append(conv1d)
            self.conv_layers.append(activation_function)
            self.conv_layers.append(max_pool)

        self.flatten = Flatten()

        dummy_input = torch.randn(1, 1, n_inputs)
        for layer in self.conv_layers:
            dummy_input = layer(dummy_input)
        dummy_input = self.flatten(dummy_input)
        prev_neurons = int(dummy_input.shape[1])

        self.dense_layers = ModuleList()
        for i, ff_dim in enumerate(feedforward_dims):
            linear = Linear(
                in_features=prev_neurons if i == 0 else feedforward_dims[i - 1],
                out_features=ff_dim,
            )
            activation_function = get_activation_function(activation)
            dropout_layer = Dropout(p=dropout)
            self.dense_layers.append(linear)
            self.dense_layers.append(activation_function)
            self.dense_layers.append(dropout_layer)

        final_in_features = prev_neurons if len(feedforward_dims) == 0 else feedforward_dims[-1]
        self.dense_layers.append(Linear(final_in_features, n_outputs))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)

        for layer in self.conv_layers:
            x = layer(x)

        x = self.flatten(x)

        for layer in self.dense_layers:
            x = layer(x)

        return x.squeeze(-1) if self.n_outputs == 1 else x
