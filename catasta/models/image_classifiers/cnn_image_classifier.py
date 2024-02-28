from torch import Tensor
from torch.nn import (
    Module,
    Conv2d,
    MaxPool2d,
    Linear,
    Flatten,
    Dropout,
    Sequential,
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
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


class CNNImageClassifier(Module):
    def __init__(self, *,
                 input_shape: tuple[int, int, int],
                 n_classes: int,
                 conv_out_channels: list[int],
                 conv_kernel_sizes: list[int],
                 conv_strides: list[int],
                 conv_paddings: list[int],
                 pooling_kernel_sizes: list[int],
                 pooling_strides: list[int],
                 pooling_paddings: list[int],
                 feedforward_dims: list[int],
                 dropout: float = 0.5,
                 activation: str = 'relu',
                 ) -> None:
        super().__init__()

        n_input_channels: int = input_shape[-1]

        conv_layers: list[Module] = []
        for i in range(len(conv_out_channels)):
            conv2d = Conv2d(
                in_channels=n_input_channels if i == 0 else conv_out_channels[i - 1],
                out_channels=conv_out_channels[i],
                kernel_size=conv_kernel_sizes[i],
                stride=conv_strides[i],
                padding=conv_paddings[i],
            )

            activation_function: Module = get_actvation_function(activation)

            max_pool2d = MaxPool2d(
                kernel_size=pooling_kernel_sizes[i],
                stride=pooling_strides[i],
                padding=pooling_paddings[i],
            )

            conv_layers.append(conv2d)
            conv_layers.append(activation_function)
            conv_layers.append(max_pool2d)

        flatten = Flatten()

        prev_neurons: int = conv_out_channels[-1] * (input_shape[0] // 2 ** len(conv_out_channels)) * (input_shape[1] // 2 ** len(conv_out_channels))
        dense_layers: list[Module] = []
        for i in range(len(feedforward_dims)):
            linear = Linear(
                in_features=prev_neurons if i == 0 else feedforward_dims[i - 1],
                out_features=feedforward_dims[i],
            )

            activation_function: Module = get_actvation_function(activation)
            dropout_layer = Dropout(p=dropout)

            dense_layers.append(linear)
            dense_layers.append(activation_function)
            dense_layers.append(dropout_layer)

        linear = Linear(
            in_features=feedforward_dims[-1],
            out_features=n_classes,
        )

        dense_layers.append(linear)

        self.net: Sequential = Sequential(*conv_layers, flatten, *dense_layers)

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, 'b h w c -> b c h w')

        return self.net(x)
