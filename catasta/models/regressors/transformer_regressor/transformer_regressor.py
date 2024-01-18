import torch
from torch import Tensor
from torch.nn import (
    Module,
    Linear,
    LayerNorm,
    Sequential,
    ModuleList,
    Dropout,
)
from torch.fft import fft

from einops import rearrange
from einops.layers.torch import Rearrange

from .modules import (
    posemb_sincos_1d,
    Attention,
    FeedForward,
)


class Transformer(Module):
    def __init__(self, *,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 head_dim: int,
                 feedforward_dim: int,
                 dropout: float,
                 ) -> None:
        super().__init__()

        self.norm = LayerNorm(d_model)
        self.layers = ModuleList([])
        for _ in range(n_layers):
            self.layers.append(ModuleList([
                Attention(d_model=d_model,
                          n_heads=n_heads,
                          head_dim=head_dim,
                          dropout=dropout,
                          ),
                FeedForward(d_model=d_model,
                            hidden_dim=feedforward_dim,
                            dropout=dropout,
                            )
            ]))

    def forward(self, x: Tensor) -> Tensor:
        for attn, ff in self.layers:  # type: ignore
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class TransformerRegressor(Module):
    def __init__(self, *,
                 context_length: int,
                 n_patches: int,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 feedforward_dim: int,
                 head_dim: int,
                 dropout: float,
                 use_fft: bool = False,
                 ) -> None:
        super().__init__()
        patch_size: int = context_length // n_patches
        freq_patch_dim: int = patch_size * 2

        if context_length % patch_size != 0:
            raise ValueError(f"sequence length {context_length} must be divisible by patch size {patch_size}")

        self.use_fft = use_fft

        self.to_patch_embedding = Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            LayerNorm(patch_size),
            Linear(patch_size, d_model),
            LayerNorm(d_model),
        )

        self.to_freq_embedding = Sequential(
            Rearrange('b c (n p) ri -> b n (p ri c)', p=patch_size),
            LayerNorm(freq_patch_dim),
            Linear(freq_patch_dim, d_model),
            LayerNorm(d_model),
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
        )

        self.linear_head = Sequential(
            LayerNorm(d_model),
            Linear(d_model, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.use_fft:
            return self.fft_forward(x)
        else:
            return self.no_fft_forward(x)

    def no_fft_forward(self, input: Tensor) -> Tensor:
        input = rearrange(input, 'b s -> b 1 s')

        x = self.to_patch_embedding(input)
        x += self.pos_embedding(x)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.linear_head(x).squeeze()

        return x

    def fft_forward(self, input: Tensor) -> Tensor:
        input = rearrange(input, 'b s -> b 1 s')
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
