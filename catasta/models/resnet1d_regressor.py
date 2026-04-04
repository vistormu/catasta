from torch import Tensor
from torch.nn import (
    AdaptiveAvgPool1d,
    BatchNorm1d,
    Conv1d,
    Dropout,
    GELU,
    Linear,
    MaxPool1d,
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


class ResidualBlock1D(Module):
    def __init__(self, *,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 activation: str,
                 ) -> None:
        super().__init__()

        self.conv1 = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = BatchNorm1d(out_channels)
        self.activation1 = get_activation_function(activation)

        self.conv2 = Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = BatchNorm1d(out_channels)
        self.activation2 = get_activation_function(activation)

        if stride != 1 or in_channels != out_channels:
            self.downsample = Sequential(
                Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                BatchNorm1d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        return self.activation2(out)


class ResNet1DRegressor(Module):
    def __init__(self, *,
                 n_inputs: int,
                 n_outputs: int,
                 channels: list[int],
                 blocks_per_stage: int = 2,
                 feedforward_dims: list[int] = [],
                 dropout: float = 0.0,
                 activation: str = 'relu',
                 ) -> None:
        super().__init__()
        self.n_outputs = n_outputs

        if n_inputs <= 0:
            raise ValueError('n_inputs must be greater than 0')
        if len(channels) == 0:
            raise ValueError('channels must contain at least one element')
        if blocks_per_stage <= 0:
            raise ValueError('blocks_per_stage must be greater than 0')

        self.stem = Sequential(
            Conv1d(1, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            BatchNorm1d(channels[0]),
            get_activation_function(activation),
            MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.stages = ModuleList()
        in_channels = channels[0]
        for stage_idx, out_channels in enumerate(channels):
            for block_idx in range(blocks_per_stage):
                stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                block = ResidualBlock1D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    activation=activation,
                )
                self.stages.append(block)
                in_channels = out_channels

        self.global_pool = AdaptiveAvgPool1d(1)

        self.head = ModuleList()
        for i, ff_dim in enumerate(feedforward_dims):
            linear = Linear(
                in_features=channels[-1] if i == 0 else feedforward_dims[i - 1],
                out_features=ff_dim,
            )
            activation_function = get_activation_function(activation)
            dropout_layer = Dropout(p=dropout)
            self.head.append(linear)
            self.head.append(activation_function)
            self.head.append(dropout_layer)

        final_in_features = channels[-1] if len(feedforward_dims) == 0 else feedforward_dims[-1]
        self.output_layer = Linear(final_in_features, n_outputs)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        elif x.ndim != 3:
            raise ValueError('input must have shape [batch, sequence] or [batch, channels, sequence]')

        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)

        x = self.global_pool(x)
        x = x.squeeze(-1)

        for layer in self.head:
            x = layer(x)

        x = self.output_layer(x)
        return x.squeeze(-1) if self.n_outputs == 1 else x
