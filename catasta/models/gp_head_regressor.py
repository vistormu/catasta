import torch
from torch import Tensor, Size
from torch.nn import Module

from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
    IndependentMultitaskVariationalStrategy,
)
from gpytorch.means import (
    ZeroMean,
    ConstantMean,
    Mean,
)
from gpytorch.kernels import (
    Kernel,
    ScaleKernel,
    RBFKernel,
    MaternKernel,
    RQKernel,
    RFFKernel,
    PeriodicKernel,
    LinearKernel,
)
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import (
    Likelihood,
    GaussianLikelihood,
    BernoulliLikelihood,
    LaplaceLikelihood,
    SoftmaxLikelihood,
    StudentTLikelihood,
    BetaLikelihood,
    MultitaskGaussianLikelihood,
)


def _get_kernel(id: str, n_inputs: int, use_ard: bool, batch_shape: Size) -> Kernel:
    ard_num_dims = n_inputs if use_ard else None
    match id.lower():
        case "rq":
            return RQKernel(ard_num_dims=ard_num_dims, batch_shape=batch_shape)
        case "matern":
            return MaternKernel(ard_num_dims=ard_num_dims, batch_shape=batch_shape)
        case "rbf":
            return RBFKernel(ard_num_dims=ard_num_dims, batch_shape=batch_shape)
        case "rff":
            return RFFKernel(num_samples=n_inputs, batch_shape=batch_shape)
        case "periodic":
            return PeriodicKernel(ard_num_dims=ard_num_dims, batch_shape=batch_shape)
        case "linear":
            return LinearKernel(ard_num_dims=ard_num_dims, batch_shape=batch_shape)
        case _:
            raise ValueError(f"Unknown kernel: {id}")


def _get_mean_module(id: str, batch_shape: Size) -> Mean:
    match id.lower():
        case "constant":
            return ConstantMean(batch_shape=batch_shape)
        case "zero":
            return ZeroMean(batch_shape=batch_shape)
        case _:
            raise ValueError(f"Unknown mean: {id}")


def _get_likelihood(id: str, n_outputs: int) -> Likelihood:
    if n_outputs > 1:
        return MultitaskGaussianLikelihood(num_tasks=n_outputs)

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


class GPHeadRegressor(ApproximateGP):
    def __init__(self, *,
                 pre_model: Module,
                 pre_model_output_dim: int,
                 n_inputs: int,
                 n_outputs: int,
                 n_inducing_points: int,
                 kernel: str = "rq",
                 mean: str = "constant",
                 likelihood: str = "gaussian",
                 use_ard: bool = True,
                 ) -> None:
        '''
        Arguments
        ---------
        pre_model: Module
            The model that will be used to preprocess the input data
        pre_model_output_dim: int
            The output dimension of the pre_model
        n_inducing_points: int
            Number of inducing points
        n_inputs: int
            Number of input dimensions
        n_outputs: int
            Number of output dimensions
        kernel: str
            Kernel to use. One of "rq", "matern", "rbf", "rff", "periodic", "linear"
        mean: str
            Mean to use. One of "constant", "zero"
        likelihood: str
            Likelihood to use. One of "gaussian", "bernoulli", "laplace", "softmax", "studentt", "beta"
        use_ard: bool
            Whether to use Automatic Relevance Determination
        '''
        if n_outputs > 1:
            raise NotImplementedError("Multi-output regression is not yet supported")

        batch_shape = Size([]) if n_outputs == 1 else Size([n_outputs])
        inducing_points_shape = (n_inducing_points, n_inputs) if n_outputs == 1 else (n_outputs, n_inducing_points, n_inputs)
        inducing_points: Tensor = torch.randn(inducing_points_shape, dtype=torch.float32)

        variational_distribution = CholeskyVariationalDistribution(
            n_inducing_points,
            batch_shape=batch_shape,
        )
        base_variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        variational_strategy = base_variational_strategy if n_outputs == 1 else IndependentMultitaskVariationalStrategy(
            base_variational_strategy=base_variational_strategy,
            num_tasks=n_outputs,
        )

        super().__init__(variational_strategy)
        self.mean_module = _get_mean_module(mean, batch_shape)
        self.covar_module = ScaleKernel(
            _get_kernel(
                kernel,
                pre_model_output_dim,
                use_ard,
                batch_shape
            ),
            batch_shape=batch_shape,
        )
        self.likelihood = _get_likelihood(likelihood, n_outputs)
        self.pre_model = pre_model

    def forward(self, x: Tensor) -> MultivariateNormal:
        x = self.pre_model(x)

        mean_x: Tensor = self.mean_module(x)  # type: ignore
        covar_x: Tensor = self.covar_module(x)  # type: ignore

        return MultivariateNormal(mean_x, covar_x)
