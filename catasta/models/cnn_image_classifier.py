import torch
from torch import Tensor
from torch.nn import (
    Conv2d,
    Module,
    ModuleList,
    MaxPool2d,
    Linear,
    Flatten,
    Dropout,
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
)

from einops import rearrange


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
            raise ValueError(f'Activation function {activation} not supported')


class CNNImageClassifier(Module):
    def __init__(self, *,
                 input_shape: tuple[int, int, int],
                 n_classes: int,
                 conv_out_channels: list[int],
                 feedforward_dims: list[int],
                 conv_kernel_size: int = 3,
                 conv_stride: int = 1,
                 conv_padding: int = 0,
                 pooling_kernel_size: int = 2,
                 pooling_stride: int = 2,
                 pooling_padding: int = 0,
                 dropout: float = 0.5,
                 activation: str = 'relu',
                 ) -> None:
        super().__init__()
        conv_kernel_sizes = [conv_kernel_size] * len(conv_out_channels)
        conv_strides = [conv_stride] * len(conv_out_channels)
        conv_paddings = [conv_padding] * len(conv_out_channels)
        pooling_kernel_sizes = [pooling_kernel_size] * len(conv_out_channels)
        pooling_strides = [pooling_stride] * len(conv_out_channels)
        pooling_paddings = [pooling_padding] * len(conv_out_channels)

        n_input_channels: int = input_shape[-1]

        # CONV LAYERS
        self.conv_layers = ModuleList()
        for i in range(len(conv_out_channels)):
            conv2d = Conv2d(
                in_channels=n_input_channels if i == 0 else conv_out_channels[i - 1],
                out_channels=conv_out_channels[i],
                kernel_size=conv_kernel_sizes[i],
                stride=conv_strides[i],
                padding=conv_paddings[i],
            )

            activation_function: Module = get_activation_function(activation)

            max_pool2d = MaxPool2d(
                kernel_size=pooling_kernel_sizes[i],
                stride=pooling_strides[i],
                padding=pooling_paddings[i],
            )

            self.conv_layers.append(conv2d)
            self.conv_layers.append(activation_function)
            self.conv_layers.append(max_pool2d)

        # FLATTEN
        self.flatten = Flatten()

        # PREVIOUS NEURONS
        dummy_input = torch.randn(input_shape)
        dummy_input = rearrange(dummy_input, 'h w c -> 1 c h w')

        for layer in self.conv_layers:
            dummy_input = layer(dummy_input)

        dummy_input = self.flatten(dummy_input)
        prev_neurons = dummy_input.shape[1]

        # DENSE LAYERS
        self.dense_layers = ModuleList()
        for i in range(len(feedforward_dims)):
            linear = Linear(
                in_features=prev_neurons if i == 0 else feedforward_dims[i - 1],
                out_features=feedforward_dims[i],
            )

            activation_function: Module = get_activation_function(activation)
            dropout_layer = Dropout(p=dropout)

            self.dense_layers.append(linear)
            self.dense_layers.append(activation_function)
            self.dense_layers.append(dropout_layer)

        linear = Linear(
            in_features=feedforward_dims[-1],
            out_features=n_classes,
        )

        self.dense_layers.append(linear)

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, 'b h w c -> b c h w')

        for layer in self.conv_layers:
            x = layer(x)

        x = self.flatten(x)

        for layer in self.dense_layers:
            x = layer(x)

        return x
