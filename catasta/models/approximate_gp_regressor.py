import torch
from torch import Tensor

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ZeroMean, ConstantMean, Mean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, RQKernel, Kernel
from gpytorch.distributions import MultivariateNormal


def _get_kernel(id: str, n_dim: int) -> Kernel | None:
    match id.lower():
        case "rq":
            return RQKernel(ard_num_dims=n_dim)
        case "matern":
            return MaternKernel(ard_num_dims=n_dim)
        case "rbf":
            return RBFKernel(ard_num_dims=n_dim)

    return None


def _get_mean_module(id: str) -> Mean | None:
    match id.lower():
        case "constant":
            return ConstantMean()
        case "zero":
            return ZeroMean()

    return None


class ApproximateGPRegressor(ApproximateGP):
    def __init__(self, n_inducing_points: int, n_dim: int, kernel: str = "rq", mean: str = "constant") -> None:
        self.n_inducing_points: int = n_inducing_points
        self.n_dim: int = n_dim
        dtype: torch.dtype = torch.float32

        inducing_points: torch.Tensor = torch.randn(n_inducing_points, n_dim, dtype=dtype)

        variational_distribution = CholeskyVariationalDistribution(n_inducing_points)
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)

        super(ApproximateGPRegressor, self).__init__(variational_strategy)

        mean_module: Mean | None = _get_mean_module(mean)
        if mean_module is None:
            raise ValueError(f"Unknown mean: {mean}")

        self.mean_module = mean_module

        kernel_module: Kernel | None = _get_kernel(kernel, n_dim)
        if kernel_module is None:
            raise ValueError(f"Unknown kernel: {kernel}")

        self.covar_module = ScaleKernel(kernel_module)

    def forward(self, x) -> MultivariateNormal:
        mean_x: Tensor = self.mean_module(x)  # type: ignore
        covar_x: Tensor = self.covar_module(x)  # type: ignore

        return MultivariateNormal(mean_x, covar_x)
