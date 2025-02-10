import torch
from torch import Tensor
from torch.distributions import Distribution

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ZeroMean, ConstantMean, Mean
from gpytorch.kernels import Kernel, ScaleKernel, RBFKernel, MaternKernel, RQKernel, RFFKernel, PeriodicKernel, LinearKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import Likelihood, GaussianLikelihood, BernoulliLikelihood, LaplaceLikelihood, SoftmaxLikelihood, StudentTLikelihood, BetaLikelihood


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


def _get_likelihood(id: str) -> Likelihood:
    match id.lower():
        case "gaussian":
            return GaussianLikelihood()
        case "bernoulli":
            return BernoulliLikelihood()
        case "laplace":
            return LaplaceLikelihood()
        case "softmax":
            return SoftmaxLikelihood()
        case "studentt":
            return StudentTLikelihood()
        case "beta":
            return BetaLikelihood()
        case _:
            raise ValueError(f"Unknown likelihood: {id}")


class GPRegressor(ApproximateGP):
    def __init__(self, *,
                 n_inducing_points: int,
                 n_inputs: int,
                 kernel: str = "rq",
                 mean: str = "constant",
                 likelihood: str = "gaussian",
                 use_ard: bool = True,
                 ) -> None:
        '''
        Arguments
        ---------
        n_inducing_points: int
            Number of inducing points
        n_inputs: int
            Number of input dimensions
        kernel: str
            Kernel to use. One of "rq", "matern", "rbf", "rff", "periodic", "linear"
        mean: str
            Mean to use. One of "constant", "zero"
        likelihood: str
            Likelihood to use. One of "gaussian", "bernoulli", "laplace", "softmax", "studentt", "beta"
        use_ard: bool
            Whether to use Automatic Relevance Determination
        '''
        inducing_points: torch.Tensor = torch.randn(n_inducing_points, n_inputs, dtype=torch.float32)

        variational_distribution = CholeskyVariationalDistribution(n_inducing_points)
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution)

        super().__init__(variational_strategy)
        self.mean_module = _get_mean_module(mean)
        self.covar_module = ScaleKernel(_get_kernel(kernel, n_inputs, use_ard))
        self.likelihood = _get_likelihood(likelihood)
        self.use_likelihood = False

    def infer(self) -> None:
        self.use_likelihood = True

    def forward(self, x: Tensor) -> MultivariateNormal | Distribution:
        mean_x: Tensor = self.mean_module(x)  # type: ignore
        covar_x: Tensor = self.covar_module(x)  # type: ignore

        latent_pred: MultivariateNormal = MultivariateNormal(mean_x, covar_x)

        if self.use_likelihood:
            return self.likelihood(latent_pred)

        return latent_pred
