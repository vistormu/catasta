import torch
from torch import nn
from torch.fft import fft
from torch import Tensor
from torch.nn import Module, Sequential, LayerNorm, Linear, GELU, Softmax

from einops import rearrange, pack, unpack, reduce
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
                 ) -> None:
        super().__init__()

        self.net = Sequential(
            LayerNorm(d_model),
            Linear(d_model, hidden_dim),
            GELU(),
            Linear(hidden_dim, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, *,
                 d_model: int,
                 n_heads: int = 8,
                 head_dim: int = 64,
                 ) -> None:
        super().__init__()

        inner_dim: int = head_dim * n_heads
        self.n_heads: int = n_heads
        self.scale: int = head_dim ** -0.5
        self.norm = LayerNorm(d_model)

        self.attend = Softmax(dim=-1)

        self.to_qkv = Linear(d_model, inner_dim * 3, bias=False)
        self.to_out = Linear(inner_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)

        qkv: Tensor = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)

        dots: Tensor = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn: Tensor = self.attend(dots)

        out: Tensor = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, *,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 head_dim: int,
                 feedforward_dim: int,
                 ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                Attention(d_model=d_model, n_heads=n_heads, head_dim=head_dim),
                FeedForward(d_model=d_model, hidden_dim=feedforward_dim)
            ]))

    def forward(self, x: Tensor) -> Tensor:
        for attn, ff in self.layers:  # type: ignore
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class FFTTransformerRegressor(Module):
    def __init__(self, *,
                 context_length: int,
                 n_patches: int,
                 output_dim: int,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 feedforward_dim: int,
                 head_dim: int,
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

        self.transformer = Transformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            feedforward_dim=feedforward_dim,
        )

        self.linear_head = nn.Linear(d_model, output_dim)

    def forward(self, input: Tensor) -> Tensor:
        input = rearrange(input, 'b s -> b 1 s')
        freqs: Tensor = torch.view_as_real(fft(input))

        x: Tensor = self.to_patch_embedding(input)
        f: Tensor = self.to_freq_embedding(freqs)

        x += posemb_sincos_1d(x)
        f += posemb_sincos_1d(f)

        x = torch.cat((x, f), dim=1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.linear_head(x).squeeze()

        return x

    def encode(self, input: Tensor) -> Tensor:
        input = rearrange(input, 'b s -> b 1 s')
        freqs: Tensor = torch.view_as_real(fft(input))

        x: Tensor = self.to_patch_embedding(input)
        f: Tensor = self.to_freq_embedding(freqs)

        x += posemb_sincos_1d(x)
        f += posemb_sincos_1d(f)

        x = torch.cat((x, f), dim=1)
        x = self.transformer(x)
        x = x.mean(dim=1)

        return x
