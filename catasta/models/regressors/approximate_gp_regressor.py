import torch
from torch import Tensor

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ZeroMean, ConstantMean, Mean
from gpytorch.kernels import Kernel, ScaleKernel, RBFKernel, MaternKernel, RQKernel, RFFKernel, PeriodicKernel
from gpytorch.distributions import MultivariateNormal


def _get_kernel(id: str, n_inputs: int) -> Kernel | None:
    match id.lower():
        case "rq":
            return RQKernel(ard_num_dims=n_inputs)
        case "matern":
            return MaternKernel(ard_num_dims=n_inputs)
        case "rbf":
            return RBFKernel(ard_num_dims=n_inputs)
        case "rff":
            return RFFKernel(num_samples=n_inputs)
        case "periodic":
            return PeriodicKernel(ard_num_dims=n_inputs)

    return None


def _get_mean_module(id: str) -> Mean | None:
    match id.lower():
        case "constant":
            return ConstantMean()
        case "zero":
            return ZeroMean()

    return None


class ApproximateGPRegressor(ApproximateGP):
    def __init__(self, *,
                 n_inducing_points: int,
                 n_inputs: int,
                 kernel: str = "rq",
                 mean: str = "constant"
                 ) -> None:
        '''
        Arguments
        ---------
        n_inducing_points: int
            Number of inducing points
        n_inputs: int
            Number of input dimensions
        kernel: str
            Kernel to use. One of "rq", "matern", "rbf", "rff", "periodic"
        mean: str
            Mean to use. One of "constant", "zero"
        '''

        self.n_inducing_points: int = n_inducing_points
        self.n_inputs: int = n_inputs
        dtype: torch.dtype = torch.float32

        inducing_points: torch.Tensor = torch.randn(n_inducing_points, n_inputs, dtype=dtype)

        variational_distribution = CholeskyVariationalDistribution(n_inducing_points)
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)

        super().__init__(variational_strategy)

        mean_module: Mean | None = _get_mean_module(mean)
        if mean_module is None:
            raise ValueError(f"Unknown mean: {mean}")

        self.mean_module = mean_module

        kernel_module: Kernel | None = _get_kernel(kernel, n_inputs)
        if kernel_module is None:
            raise ValueError(f"Unknown kernel: {kernel}")

        self.covar_module = ScaleKernel(kernel_module)

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x: Tensor = self.mean_module(x)  # type: ignore
        covar_x: Tensor = self.covar_module(x)  # type: ignore

        return MultivariateNormal(mean_x, covar_x)
