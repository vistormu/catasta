import torch
from torch import Tensor

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ZeroMean, ConstantMean, Mean
from gpytorch.kernels import Kernel, ScaleKernel, RBFKernel, MaternKernel, RQKernel, RFFKernel, PeriodicKernel, LinearKernel
from gpytorch.distributions import MultivariateNormal


def _get_kernel(id: str, context_length: int) -> Kernel:
    match id.lower():
        case "rq":
            return RQKernel(ard_num_dims=context_length)
        case "matern":
            return MaternKernel(ard_num_dims=context_length)
        case "rbf":
            return RBFKernel(ard_num_dims=context_length)
        case "rff":
            return RFFKernel(num_samples=context_length)
        case "periodic":
            return PeriodicKernel(ard_num_dims=context_length)
        case "linear":
            return LinearKernel(ard_num_dims=context_length)
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


class ApproximateGPRegressor(ApproximateGP):
    def __init__(self, *,
                 n_inducing_points: int,
                 context_length: int,
                 kernel: str = "rq",
                 mean: str = "constant"
                 ) -> None:
        '''
        Arguments
        ---------
        n_inducing_points: int
            Number of inducing points
        context_length: int
            Number of input dimensions
        kernel: str
            Kernel to use. One of "rq", "matern", "rbf", "rff", "periodic", "linear"
        mean: str
            Mean to use. One of "constant", "zero"
        '''
        inducing_points: torch.Tensor = torch.randn(n_inducing_points, context_length, dtype=torch.float32)

        variational_distribution = CholeskyVariationalDistribution(n_inducing_points)
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)

        super().__init__(variational_strategy)
        self.mean_module = _get_mean_module(mean)
        self.covar_module = ScaleKernel(_get_kernel(kernel, context_length))

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x: Tensor = self.mean_module(x)  # type: ignore
        covar_x: Tensor = self.covar_module(x)  # type: ignore

        return MultivariateNormal(mean_x, covar_x)
