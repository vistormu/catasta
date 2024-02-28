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
from torch.fft import fft2

from einops import rearrange
from einops.layers.torch import Rearrange


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


class TransformerImageClassifier(Module):
    def __init__(self, *,
                 input_shape: tuple[int, int, int],
                 n_classes: int,
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
            Linear(d_model, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fft_forward(x) if self.use_fft else self.no_fft_forward(x)

    def no_fft_forward(self, input: Tensor) -> Tensor:
        x = self.to_patch_embedding(input)
        x += self.pos_embedding.to(x.device)

        x = self.transformer(x)
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

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.linear_head(x)

        return x
