import torch
from torch import Tensor
from torch.distributions import Bernoulli

from gpytorch.models import ApproximateGP
from gpytorch.variational import NaturalVariationalDistribution, VariationalStrategy
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RQKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _OneDimensionalLikelihood


class PGLikelihood(_OneDimensionalLikelihood):
    def expected_log_prob(self, target, input, *args, **kwargs):
        mean, variance = input.mean, input.variance
        raw_second_moment = variance + mean.pow(2)
        target = target.to(mean.dtype).mul(2.).sub(1.)
        c = raw_second_moment.detach().sqrt()
        half_omega = 0.25 * torch.tanh(0.5 * c) / c
        res = 0.5 * target * mean - half_omega * raw_second_moment
        res = res.sum(dim=-1)

        return res

    def forward(self, function_samples):
        return Bernoulli(logits=function_samples)

    def marginal(self, function_dist):
        def prob_lambda(function_samples): return self.forward(function_samples).probs
        probs = self.quadrature(prob_lambda, function_dist)
        return Bernoulli(probs=probs)


class GPModel(ApproximateGP):
    def __init__(self, inducing_points: Tensor, n_dimensions: int) -> None:
        variational_distribution = NaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)

        super(GPModel, self).__init__(variational_strategy)

        self.mean_module: ZeroMean = ZeroMean()
        self.covar_module: ScaleKernel = ScaleKernel(RQKernel(ard_num_dims=n_dimensions))

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x: Tensor = self.mean_module(x)  # type: ignore
        covar_x: Tensor = self.covar_module(x)  # type: ignore

        return MultivariateNormal(mean_x, covar_x)
