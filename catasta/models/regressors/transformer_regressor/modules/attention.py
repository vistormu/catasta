import torch
from torch import Tensor
from torch.nn import (
    Module,
    Linear,
    LayerNorm,
    Softmax,
    Sequential,
    Dropout,
)

from einops import rearrange


class Attention(Module):
    def __init__(self, *,
                 d_model: int,
                 n_heads: int,
                 head_dim: int,
                 dropout: float,
                 ) -> None:
        super().__init__()

        self.n_heads: int = n_heads
        self.scale: int = head_dim ** -0.5
        self.norm = LayerNorm(d_model)
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
