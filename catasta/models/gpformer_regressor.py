from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, ScaleKernel, RBFKernel, MaternKernel, RQKernel, RFFKernel, PeriodicKernel, LinearKernel
from gpytorch.means import ZeroMean, ConstantMean, Mean
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.models import ApproximateGP

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

from einops import rearrange, reduce
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


class TransformerFeatureExtractor(Module):
    def __init__(self, *,
                 n_inputs: int,
                 n_patches: int,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 feedforward_dim: int,
                 head_dim: int,
                 pooling: str,
                 dropout: float,
                 layer_norm: bool,
                 ) -> None:
        super().__init__()
        patch_size: int = n_inputs // n_patches

        if n_inputs % patch_size != 0:
            raise ValueError(f"sequence length {n_inputs} must be divisible by patch size {patch_size}")

        self.to_patch_embedding = Sequential(
            Rearrange('b (n p) -> b n p', p=patch_size),
            Linear(patch_size, d_model),
        )

        self.pos_embedding = posemb_sincos_1d
        self.dropout = Dropout(dropout)

        self.pooling: str = pooling

        self.transformer = Transformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.to_patch_embedding(x)
        x += self.pos_embedding(x)

        x = self.transformer(x)
        x = self.dropout(x)

        x = reduce(x, 'b n d -> b d', self.pooling) if self.pooling != "concat" else rearrange(x, 'b n d -> b (n d)')

        return x


def _get_kernel(id: str, n_inputs: int, use_ard: bool) -> Kernel:
    match id.lower():
        case "rq":
            return RQKernel(ard_num_dims=n_inputs if use_ard else None)
        case "matern":
            return MaternKernel(ard_num_dims=n_inputs if use_ard else None)
        case "rbf":
            return RBFKernel(ard_num_dims=n_inputs if use_ard else None)
        case "rff":
            return RFFKernel(num_samples=n_inputs)
        case "periodic":
            return PeriodicKernel(ard_num_dims=n_inputs if use_ard else None)
        case "linear":
            return LinearKernel(ard_num_dims=n_inputs if use_ard else None)
        case _:
            raise ValueError(f"Unknown kernel: {id}")


def _get_mean_module(id: str) -> Mean:
    match id.lower():
        case "constant":
            return ConstantMean()
        case "zero":
            return ZeroMean()
        case _:
            raise ValueError(f"Unknown mean: {id}")


class GPFormerRegressor(ApproximateGP):
    def __init__(self, *,
                 n_inducing_points: int,
                 n_inputs: int,
                 n_patches: int,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 feedforward_dim: int,
                 head_dim: int,
                 dropout: float,
                 layer_norm: bool = False,
                 kernel: str = "rq",
                 mean: str = "constant",
                 pooling: str = "mean",
                 use_ard: bool = True,
                 ) -> None:

        input_dim: int = d_model * n_patches if pooling == "concat" else d_model

        inducing_points: torch.Tensor = torch.randn(n_inducing_points, n_inputs, dtype=torch.float32)

        variational_distribution = CholeskyVariationalDistribution(n_inducing_points)
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution)

        super().__init__(variational_strategy)
        self.mean_module = _get_mean_module(mean)
        self.covar_module = ScaleKernel(_get_kernel(kernel, input_dim, use_ard))

        self.feature_extractor = TransformerFeatureExtractor(
            n_inputs=n_inputs,
            n_patches=n_patches,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            feedforward_dim=feedforward_dim,
            head_dim=head_dim,
            pooling=pooling,
            dropout=dropout,
            layer_norm=layer_norm,
        )

    def forward(self, x: Tensor) -> MultivariateNormal:
        x = self.feature_extractor(x)

        mean_x: Tensor = self.mean_module(x)  # type: ignore
        covar_x: Tensor = self.covar_module(x)  # type: ignore

        return MultivariateNormal(mean_x, covar_x)
