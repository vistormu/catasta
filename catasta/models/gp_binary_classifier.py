from __future__ import annotations

import torch
from torch import Size, Tensor
from torch.distributions import Bernoulli

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import (
    Kernel,
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    RBFKernel,
    RFFKernel,
    RQKernel,
    ScaleKernel,
)
from gpytorch.likelihoods import BernoulliLikelihood, Likelihood, _OneDimensionalLikelihood
from gpytorch.means import ConstantMean, Mean, ZeroMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import NaturalVariationalDistribution, VariationalStrategy

KernelSpec = str | type[Kernel] | Kernel
MeanSpec = str | type[Mean] | Mean
LikelihoodSpec = str | type[Likelihood] | Likelihood


def _instantiate_with_fallbacks(class_type: type, kwargs_options: list[dict]) -> object:
    for kwargs in kwargs_options:
        try:
            return class_type(**kwargs)
        except TypeError:
            continue
    raise ValueError(f"could not instantiate class {class_type.__name__} with supported constructor patterns")


def _get_kernel(spec: KernelSpec, n_inputs: int, use_ard: bool, batch_shape: Size) -> Kernel:
    ard_num_dims = n_inputs if use_ard else None

    if isinstance(spec, Kernel):
        return spec

    if isinstance(spec, type):
        if not issubclass(spec, Kernel):
            raise ValueError("kernel class must inherit gpytorch.kernels.Kernel")
        return _instantiate_with_fallbacks(
            spec,
            [
                {"ard_num_dims": ard_num_dims, "batch_shape": batch_shape},
                {"batch_shape": batch_shape},
                {"ard_num_dims": ard_num_dims},
                {},
            ],
        )  # type: ignore[return-value]

    match spec.lower():
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
            raise ValueError(f"unknown kernel: {spec}")


def _get_mean_module(spec: MeanSpec, batch_shape: Size) -> Mean:
    if isinstance(spec, Mean):
        return spec

    if isinstance(spec, type):
        if not issubclass(spec, Mean):
            raise ValueError("mean class must inherit gpytorch.means.Mean")
        return _instantiate_with_fallbacks(spec, [{"batch_shape": batch_shape}, {}])  # type: ignore[return-value]

    match spec.lower():
        case "zero":
            return ZeroMean(batch_shape=batch_shape)
        case "constant":
            return ConstantMean(batch_shape=batch_shape)
        case _:
            raise ValueError(f"unknown mean: {spec}")


class PGLikelihood(_OneDimensionalLikelihood):
    def expected_log_prob(self, target: Tensor, input: MultivariateNormal, *args, **kwargs) -> Tensor:
        mean, variance = input.mean, input.variance
        raw_second_moment = variance + mean.pow(2)

        target = target.to(mean.dtype).mul(2.0).sub(1.0)

        c = raw_second_moment.detach().sqrt().clamp_min(1e-8)
        half_omega = 0.25 * torch.tanh(0.5 * c) / c

        res = 0.5 * target * mean - half_omega * raw_second_moment
        return res.sum(dim=-1)

    def forward(self, function_samples: Tensor, *args, **kwargs) -> Bernoulli:
        return Bernoulli(logits=function_samples)

    def marginal(self, function_dist: MultivariateNormal, *args, **kwargs) -> Bernoulli:
        def probability_fn(function_samples: Tensor) -> Tensor:
            return self.forward(function_samples).probs

        probs = self.quadrature(probability_fn, function_dist)
        return Bernoulli(probs=probs)


def _get_likelihood(spec: LikelihoodSpec) -> Likelihood:
    if isinstance(spec, Likelihood):
        return spec

    if isinstance(spec, type):
        if not issubclass(spec, Likelihood):
            raise ValueError("likelihood class must inherit gpytorch.likelihoods.Likelihood")
        return _instantiate_with_fallbacks(spec, [{}])  # type: ignore[return-value]

    match spec.lower():
        case "pg" | "polya_gamma" | "polya-gamma":
            return PGLikelihood()
        case "bernoulli":
            return BernoulliLikelihood()
        case _:
            raise ValueError(f"unknown likelihood: {spec}")


class GPBinaryClassifier(ApproximateGP):
    def __init__(self, *,
                 n_inducing_points: int,
                 n_inputs: int,
                 kernel: KernelSpec = "rq",
                 mean: MeanSpec = "zero",
                 likelihood: LikelihoodSpec = "pg",
                 use_ard: bool = True,
                 ) -> None:
        batch_shape = Size([])
        inducing_points = torch.randn(n_inducing_points, n_inputs, dtype=torch.float32)

        variational_distribution = NaturalVariationalDistribution(n_inducing_points)
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy)

        self.mean_module = _get_mean_module(mean, batch_shape)
        self.covar_module = ScaleKernel(
            _get_kernel(kernel, n_inputs, use_ard, batch_shape),
            batch_shape=batch_shape,
        )
        self.likelihood = _get_likelihood(likelihood)

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x: Tensor = self.mean_module(x)  # type: ignore[assignment]
        covar_x: Tensor = self.covar_module(x)  # type: ignore[assignment]
        return MultivariateNormal(mean_x, covar_x)
