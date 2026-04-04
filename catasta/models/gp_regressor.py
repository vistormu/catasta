from __future__ import annotations

from typing import TypeAlias

import torch
from torch import Size, Tensor

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
from gpytorch.likelihoods import (
    BernoulliLikelihood,
    BetaLikelihood,
    GaussianLikelihood,
    LaplaceLikelihood,
    Likelihood,
    MultitaskGaussianLikelihood,
    SoftmaxLikelihood,
    StudentTLikelihood,
)
from gpytorch.means import ConstantMean, Mean, ZeroMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)

KernelSpec: TypeAlias = str | type[Kernel] | Kernel
MeanSpec: TypeAlias = str | type[Mean] | Mean
LikelihoodSpec: TypeAlias = str | type[Likelihood] | Likelihood


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
            raise ValueError(f"Unknown kernel: {spec}")


def _get_mean_module(spec: MeanSpec, batch_shape: Size) -> Mean:
    if isinstance(spec, Mean):
        return spec

    if isinstance(spec, type):
        if not issubclass(spec, Mean):
            raise ValueError("mean class must inherit gpytorch.means.Mean")
        return _instantiate_with_fallbacks(spec, [{"batch_shape": batch_shape}, {}])  # type: ignore[return-value]

    match spec.lower():
        case "constant":
            return ConstantMean(batch_shape=batch_shape)
        case "zero":
            return ZeroMean(batch_shape=batch_shape)
        case _:
            raise ValueError(f"Unknown mean: {spec}")


def _get_likelihood(spec: LikelihoodSpec, n_outputs: int) -> Likelihood:
    if isinstance(spec, Likelihood):
        return spec

    if isinstance(spec, type):
        if not issubclass(spec, Likelihood):
            raise ValueError("likelihood class must inherit gpytorch.likelihoods.Likelihood")
        if n_outputs > 1:
            return _instantiate_with_fallbacks(spec, [{"num_tasks": n_outputs}, {}])  # type: ignore[return-value]
        return _instantiate_with_fallbacks(spec, [{}])  # type: ignore[return-value]

    if n_outputs > 1:
        return MultitaskGaussianLikelihood(num_tasks=n_outputs)

    match spec.lower():
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
            raise ValueError(f"Unknown likelihood: {spec}")


class GPRegressor(ApproximateGP):
    def __init__(
        self,
        *,
        n_inducing_points: int,
        n_inputs: int,
        n_outputs: int,
        kernel: KernelSpec = "rq",
        mean: MeanSpec = "constant",
        likelihood: LikelihoodSpec = "gaussian",
        use_ard: bool = True,
    ) -> None:
        batch_shape = Size([]) if n_outputs == 1 else Size([n_outputs])
        inducing_points_shape = (
            (n_inducing_points, n_inputs)
            if n_outputs == 1
            else (n_outputs, n_inducing_points, n_inputs)
        )
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
        variational_strategy = (
            base_variational_strategy
            if n_outputs == 1
            else IndependentMultitaskVariationalStrategy(
                base_variational_strategy=base_variational_strategy,
                num_tasks=n_outputs,
            )
        )

        super().__init__(variational_strategy)
        self.mean_module = _get_mean_module(mean, batch_shape)
        self.covar_module = ScaleKernel(
            _get_kernel(kernel, n_inputs, use_ard, batch_shape),
            batch_shape=batch_shape,
        )
        self.likelihood = _get_likelihood(likelihood, n_outputs)

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x: Tensor = self.mean_module(x)  # type: ignore[assignment]
        covar_x: Tensor = self.covar_module(x)  # type: ignore[assignment]
        return MultivariateNormal(mean_x, covar_x)
