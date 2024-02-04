import torch
from torch import Tensor
from torch.nn import (
    Module,
    Linear,
    LayerNorm,
    Sequential,
    ModuleList,
    Dropout,
    Softmax,
    GELU,
    Identity,
)
from torch.fft import fft

from einops import rearrange
from einops.layers.torch import Rearrange


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


class FeedForward(Module):
    def __init__(self, *,
                 d_model: int,
                 hidden_dim: int,
                 dropout: float,
                 layer_norm: bool,
                 ) -> None:
        super().__init__()

        self.net = Sequential(
            LayerNorm(d_model) if layer_norm else Identity(),
            Linear(d_model, hidden_dim),
            GELU(),
            Dropout(dropout),
            Linear(hidden_dim, d_model),
            Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(Module):
    def __init__(self, *,
                 d_model: int,
                 n_heads: int,
                 head_dim: int,
                 dropout: float,
                 layer_norm: bool,
                 ) -> None:
        super().__init__()

        self.n_heads: int = n_heads
        self.scale: int = head_dim ** -0.5
        self.norm = LayerNorm(d_model) if layer_norm else Identity()
        self.attend = Softmax(dim=-1)
        self.dropout = Dropout(dropout)

        inner_dim: int = head_dim * n_heads
        self.to_qkv = Linear(d_model, inner_dim * 3, bias=False)
        self.to_out = Sequential(
            Linear(inner_dim, d_model, bias=False),
            Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)

        qkv: Tensor = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)

        dots: Tensor = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn: Tensor = self.attend(dots)
        attn = self.dropout(attn)

        out: Tensor = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Transformer(Module):
    def __init__(self, *,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 head_dim: int,
                 feedforward_dim: int,
                 dropout: float,
                 layer_norm: bool,
                 ) -> None:
        super().__init__()

        self.norm = LayerNorm(d_model) if layer_norm else Identity()
        self.layers = ModuleList([])
        for _ in range(n_layers):
            self.layers.append(ModuleList([
                Attention(d_model=d_model,
                          n_heads=n_heads,
                          head_dim=head_dim,
                          dropout=dropout,
                          layer_norm=layer_norm,
                          ),
                FeedForward(d_model=d_model,
                            hidden_dim=feedforward_dim,
                            dropout=dropout,
                            layer_norm=layer_norm,
                            )
            ]))

    def forward(self, x: Tensor) -> Tensor:
        for attn, ff in self.layers:  # type: ignore
            x = attn(x) + x
            x = ff(x) + x

        x = self.norm(x)

        return x


class TransformerRegressor(Module):
    def __init__(self, *,
                 context_length: int,
                 n_patches: int,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 feedforward_dim: int,
                 head_dim: int,
                 dropout: float = 0.0,
                 layer_norm: bool = False,
                 use_fft: bool = False,
                 ) -> None:
        super().__init__()
        patch_size: int = context_length // n_patches
        freq_patch_dim: int = patch_size * 2

        if context_length % patch_size != 0:
            raise ValueError(f"sequence length {context_length} must be divisible by patch size {patch_size}")

        self.use_fft = use_fft

        self.to_patch_embedding = Sequential(
            Rearrange('b (n p) -> b n p', p=patch_size),
            LayerNorm(patch_size) if layer_norm else Identity(),
            Linear(patch_size, d_model),
            LayerNorm(d_model) if layer_norm else Identity(),
        )

        self.to_freq_embedding = Sequential(
            Rearrange('b (n p) ri -> b n (p ri)', p=patch_size),
            LayerNorm(freq_patch_dim) if layer_norm else Identity(),
            Linear(freq_patch_dim, d_model),
            LayerNorm(d_model) if layer_norm else Identity(),
        )

        self.pos_embedding = posemb_sincos_1d
        self.dropout = Dropout(dropout)

        self.transformer = Transformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            layer_norm=layer_norm,
        )

        self.linear_head = Sequential(
            LayerNorm(d_model) if layer_norm else Identity(),
            Linear(d_model, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.use_fft:
            return self.fft_forward(x)
        else:
            return self.no_fft_forward(x)

    def no_fft_forward(self, input: Tensor) -> Tensor:
        x = self.to_patch_embedding(input)
        x += self.pos_embedding(x)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.linear_head(x).squeeze()

        return x

    def fft_forward(self, input: Tensor) -> Tensor:
        freqs: Tensor = torch.view_as_real(fft(input))

        x = self.to_patch_embedding(input)
        f = self.to_freq_embedding(freqs)

        x += self.pos_embedding(x)
        f += self.pos_embedding(f)

        x = torch.cat((x, f), dim=1)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.linear_head(x).squeeze()

        return x
