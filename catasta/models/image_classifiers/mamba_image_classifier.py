import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

from torch import Tensor
from torch.nn import (
    Module,
    Linear,
    LayerNorm,
    Sequential,
    ModuleList,
    Identity,
)
from torch.fft import fft2

from einops.layers.torch import Rearrange


class ResidualBlock(Module):
    def __init__(self, *,
                 d_model: int,
                 d_inner: int,
                 conv_bias: bool,
                 d_conv: int,
                 bias: bool,
                 d_state: int,
                 dt_rank: int,
                 ) -> None:
        super().__init__()
        self.mixer = MambaBlock(
            d_model=d_model,
            d_inner=d_inner,
            conv_bias=conv_bias,
            d_conv=d_conv,
            bias=bias,
            d_state=d_state,
            dt_rank=dt_rank,
        )
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x

        return output


class MambaBlock(Module):
    def __init__(self, *,
                 d_model: int,
                 d_inner: int,
                 conv_bias: bool,
                 d_conv: int,
                 bias: bool,
                 d_state: int,
                 dt_rank: int,
                 ) -> None:
        super().__init__()

        self.d_inner = d_inner
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)

        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


def posemb_sincos_2d(h: int, w: int, dim: int, temperature: int = 10000) -> Tensor:
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

    if dim % 4 != 0:
        raise ValueError(f"feature dimension {dim} must be multiple of 4 for 2D sin-cos positional embedding")

    omega: Tensor = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)

    return pe.to(torch.float32)


class MambaImageClassifier(Module):
    def __init__(self, *,
                 input_shape: tuple[int, int, int],
                 n_classes: int,
                 n_patches: int,
                 d_model: int,
                 d_state: int,
                 d_conv: int,
                 expand: int,
                 n_layers: int,
                 layer_norm: bool = False,
                 use_fft: bool = False,
                 conv_bias: bool = False,
                 bias: bool = False,
                 ) -> None:
        super().__init__()

        image_height: int = input_shape[0]
        image_width: int = input_shape[1]
        image_channels: int = input_shape[2]

        patch_height: int = image_height // n_patches
        patch_width: int = image_width // n_patches
        patch_size: int = patch_height * patch_width * image_channels

        freq_patch_size: int = patch_size * 2

        if image_height % n_patches != 0 or image_width % n_patches != 0:
            raise ValueError(f"image size ({input_shape[0]}, {input_shape[1]}) must be divisible by number of patches {n_patches}")

        self.use_fft = use_fft

        self.to_patch_embedding = Sequential(
            Rearrange('b (h ph) (w pw) c -> b (h w) (ph pw c)', ph=patch_height, pw=patch_width),
            LayerNorm(patch_size) if layer_norm else Identity(),
            Linear(patch_size, d_model),
            LayerNorm(d_model) if layer_norm else Identity(),
        )

        self.to_freq_embedding = Sequential(
            Rearrange('b (h ph) (w pw) c ri -> b (h w) (ph pw c ri)', ph=patch_height, pw=patch_width),
            LayerNorm(freq_patch_size) if layer_norm else Identity(),
            Linear(freq_patch_size, d_model),
            LayerNorm(d_model) if layer_norm else Identity(),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=d_model,
        )

        self.layers: ModuleList = ModuleList([
            ResidualBlock(
                d_model=d_model,
                d_inner=d_model * expand,
                conv_bias=conv_bias,
                d_conv=d_conv,
                bias=bias,
                d_state=d_state,
                dt_rank=math.ceil(d_model / 16),
            )
            for _ in range(n_layers)
        ])

        self.norm_f = RMSNorm(d_model) if layer_norm else Identity()

        self.linear_head = Sequential(
            LayerNorm(d_model) if layer_norm else Identity(),
            Linear(d_model, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fft_forward(x) if self.use_fft else self.no_fft_forward(x)

    def no_fft_forward(self, input: Tensor) -> Tensor:
        x = self.to_patch_embedding(input)
        x += self.pos_embedding.to(x.device)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)

        x = x.mean(dim=1)

        x = self.linear_head(x)

        return x

    def fft_forward(self, input: Tensor) -> Tensor:
        freqs: Tensor = torch.view_as_real(fft2(input))

        x = self.to_patch_embedding(input)
        f = self.to_freq_embedding(freqs)

        x += self.pos_embedding.to(x.device)
        f += self.pos_embedding.to(f.device)

        x = torch.cat((x, f), dim=1)
        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        x = x.mean(dim=1)

        x = self.linear_head(x)

        return x
