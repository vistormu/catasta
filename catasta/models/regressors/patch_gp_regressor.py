import torch
from torch import Tensor
from torch.nn import Linear, Sequential
from einops.layers.torch import Rearrange, Reduce

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ZeroMean, ConstantMean, Mean
from gpytorch.kernels import Kernel, ScaleKernel, RBFKernel, MaternKernel, RQKernel, RFFKernel, PeriodicKernel, LinearKernel
from gpytorch.distributions import MultivariateNormal


def _get_kernel(id: str, context_length: int, use_ard: bool) -> Kernel:
    match id.lower():
        case "rq":
            return RQKernel(ard_num_dims=context_length if use_ard else None)
        case "matern":
            return MaternKernel(ard_num_dims=context_length if use_ard else None)
        case "rbf":
            return RBFKernel(ard_num_dims=context_length if use_ard else None)
        case "rff":
            return RFFKernel(num_samples=context_length)
        case "periodic":
            return PeriodicKernel(ard_num_dims=context_length if use_ard else None)
        case "linear":
            return LinearKernel(ard_num_dims=context_length if use_ard else None)
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


class PatchGPRegressor(ApproximateGP):
    def __init__(self, *,
                 n_inducing_points: int,
                 context_length: int,
                 n_patches: int,
                 d_model: int,
                 kernel: str = "rq",
                 mean: str = "constant",
                 pooling: str = "concat",
                 use_ard: bool = True,
                 ) -> None:
        '''
        Arguments
        ---------
        n_inducing_points: int
            Number of inducing points
        context_length: int
            Number of input dimensions
        n_patches: int
            Number of patches to divide the context length into
        d_model: int
            Dimensionality of the resulting vectors after the patches are embedded
        kernel: str
            Kernel to use. One of "rq", "matern", "rbf", "rff", "periodic", "linear"
        mean: str
            Mean to use. One of "constant", "zero"
        pooling: str
            How to pool the patches. One of "concat", "sum", "mean".
        use_ard: bool
            Whether to use Automatic Relevance Determination
        '''
        inducing_points: torch.Tensor = torch.randn(n_inducing_points, context_length, dtype=torch.float32)

        variational_distribution = CholeskyVariationalDistribution(n_inducing_points)
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution)

        super().__init__(variational_strategy)

        patch_output_dim: int = d_model * n_patches if pooling == "concat" else d_model

        self.mean_module = _get_mean_module(mean)
        self.covar_module = ScaleKernel(_get_kernel(kernel, patch_output_dim, use_ard))

        patch_size: int = context_length // n_patches

        pool_layer = Rearrange('b n d -> b (n d)') if pooling == "concat" else Reduce('b n d -> b d', 'mean' if pooling == "mean" else 'sum')

        self.to_patch_embedding = Sequential(
            Rearrange('b (n p) -> b n p', p=patch_size),
            Linear(patch_size, d_model),
            pool_layer,
        )

    def forward(self, x: Tensor) -> MultivariateNormal:
        x = self.to_patch_embedding(x)

        mean_x: Tensor = self.mean_module(x)  # type: ignore
        covar_x: Tensor = self.covar_module(x)  # type: ignore

        return MultivariateNormal(mean_x, covar_x)
