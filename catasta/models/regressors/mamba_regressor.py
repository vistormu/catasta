import torch
from einops.layers.torch import Rearrange
from einops import rearrange
from torch.nn import Module, Sequential, LayerNorm, Linear
from torch import Tensor
from torch.nn import Module

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


class MambaRegressor(Module):
    def __init__(self, *,
                 context_length: int,
                 n_patches: int,
                 d_model: int,
                 d_state: int,
                 d_conv: int,
                 expand: int,
                 ) -> None:
        super().__init__()
        patch_size: int = context_length // n_patches

        if context_length % patch_size != 0:
            raise ValueError(f"sequence length {context_length} must be divisible by patch size {patch_size}")

        self.to_patch_embedding = Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            LayerNorm(patch_size),
            Linear(patch_size, d_model),
            LayerNorm(d_model),
        )

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.linear_head = Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, 'b s -> b 1 s')

        x = self.to_patch_embedding(x)
        x += posemb_sincos_1d(x)

        x = self.mamba(x)
        x = x.mean(dim=1)

        x = self.linear_head(x).squeeze(1)

        return x
