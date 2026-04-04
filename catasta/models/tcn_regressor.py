from torch import Tensor
from torch.nn import (
    Conv1d,
    Dropout,
    GELU,
    Linear,
    Module,
    ModuleList,
    ReLU,
    Sequential,
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


class Chomp1d(Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: Tensor) -> Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(Module):
    def __init__(self, *,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int,
                 dropout: float,
                 activation: str,
                 ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.activation1 = get_activation_function(activation)
        self.dropout1 = Dropout(dropout)

        self.conv2 = Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.activation2 = get_activation_function(activation)
        self.dropout2 = Dropout(dropout)

        self.downsample = Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.out_activation = get_activation_function(activation)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.activation1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.activation2(out)
        out = self.dropout2(out)

        residual = x if self.downsample is None else self.downsample(x)
        return self.out_activation(out + residual)


class TCNRegressor(Module):
    def __init__(self, *,
                 n_inputs: int,
                 n_outputs: int,
                 channel_sizes: list[int],
                 kernel_size: int = 3,
                 feedforward_dims: list[int] = [],
                 dropout: float = 0.0,
                 activation: str = 'relu',
                 ) -> None:
        super().__init__()
        self.n_outputs = n_outputs

        if n_inputs <= 0:
            raise ValueError('n_inputs must be greater than 0')
        if len(channel_sizes) == 0:
            raise ValueError('channel_sizes must contain at least one element')

        self.tcn_blocks = ModuleList()
        in_channels = 1
        for i, out_channels in enumerate(channel_sizes):
            dilation = 2 ** i
            block = TemporalBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
                activation=activation,
            )
            self.tcn_blocks.append(block)
            in_channels = out_channels

        self.head = ModuleList()
        for i, ff_dim in enumerate(feedforward_dims):
            linear = Linear(
                in_features=channel_sizes[-1] if i == 0 else feedforward_dims[i - 1],
                out_features=ff_dim,
            )
            activation_function = get_activation_function(activation)
            dropout_layer = Dropout(p=dropout)
            self.head.append(linear)
            self.head.append(activation_function)
            self.head.append(dropout_layer)

        final_in_features = channel_sizes[-1] if len(feedforward_dims) == 0 else feedforward_dims[-1]
        self.output_layer = Linear(final_in_features, n_outputs)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError('input must have shape [batch, sequence] or [batch, channels, sequence]')

        for block in self.tcn_blocks:
            x = block(x)

        x = x[:, :, -1]

        for layer in self.head:
            x = layer(x)

        x = self.output_layer(x)
        return x.squeeze(-1) if self.n_outputs == 1 else x
