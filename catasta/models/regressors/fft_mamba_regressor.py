import torch
from torch import nn
from torch.fft import fft
from torch import Tensor
from torch.nn import Module, Sequential, LayerNorm, Linear, GELU, Softmax

from einops import rearrange
from einops.layers.torch import Rearrange

from mamba_ssm import Mamba


def posemb_sincos_1d(patches: Tensor, temperature: int = 10000) -> Tensor:
    n: int = patches.shape[1]
    d_model: int = patches.shape[2]
    device: torch.device = patches.device
    dtype: torch.dtype = patches.dtype

    if (d_model % 2) != 0:
        raise ValueError(f'feature dimension must be multiple of 2 for sincos emb. got {d_model}')

    n_tensor: Tensor = torch.arange(n, device=device)
    omega: Tensor = torch.arange(d_model // 2, device=device) / (d_model // 2 - 1)
    omega = 1.0 / (temperature ** omega)

    n_tensor = n_tensor.flatten()[:, None] * omega[None, :]
    pe: Tensor = torch.cat((n_tensor.sin(), n_tensor.cos()), dim=1)

    return pe.to(dtype)


class FFTMambaRegressor(Module):
    def __init__(self, *,
                 context_length: int,
                 n_patches: int,
                 d_model: int,
                 d_state: int,
                 d_conv: int,
                 expand: int,
                 ) -> None:
        super().__init__()
        patch_dim: int = context_length // n_patches
        freq_patch_dim: int = patch_dim * 2

        if context_length % patch_dim != 0:
            raise ValueError(f"sequence length {context_length} must be divisible by patch size {patch_dim}")

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_dim),
            LayerNorm(patch_dim),
            Linear(patch_dim, d_model),
            LayerNorm(d_model),
        )

        self.to_freq_embedding = nn.Sequential(
            Rearrange('b c (n p) ri -> b n (p ri c)', p=patch_dim),
            LayerNorm(freq_patch_dim),
            Linear(freq_patch_dim, d_model),
            LayerNorm(d_model),
        )

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.linear_head = nn.Linear(d_model, 1)

    def forward(self, input: Tensor) -> Tensor:
        input = rearrange(input, 'b s -> b 1 s')
        freqs: Tensor = torch.view_as_real(fft(input))

        x: Tensor = self.to_patch_embedding(input)
        f: Tensor = self.to_freq_embedding(freqs)

        x += posemb_sincos_1d(x)
        f += posemb_sincos_1d(f)

        x = torch.cat((x, f), dim=1)
        x = self.mamba(x)
        x = x.mean(dim=1)

        x = self.linear_head(x).squeeze()

        return x
