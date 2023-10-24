import torch
from torch import Tensor

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal


class ApproximateGPRegressor(ApproximateGP):
    def __init__(self, n_inducing_points: int, n_dim: int) -> None:
        self.n_inducing_points: int = n_inducing_points
        self.n_dim: int = n_dim
        dtype: torch.dtype = torch.float32

        inducing_points: torch.Tensor = torch.randn(n_inducing_points, n_dim, dtype=dtype)

        variational_distribution = CholeskyVariationalDistribution(n_inducing_points)
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)

        super(ApproximateGPRegressor, self).__init__(variational_strategy)

        self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=n_dim))

    def forward(self, x) -> MultivariateNormal:
        mean_x: Tensor = self.mean_module(x)  # type: ignore
        covar_x: Tensor = self.covar_module(x)  # type: ignore

        return MultivariateNormal(mean_x, covar_x)
