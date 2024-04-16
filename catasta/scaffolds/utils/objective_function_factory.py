from torch.nn import Module

from gpytorch.mlls import (
    VariationalELBO,
    PredictiveLogLikelihood,
    VariationalMarginalLogLikelihood,
    GammaRobustVariationalELBO,
)


def get_objective_function(id: str | Module, model: Module, likelihood: Module, num_data: int) -> Module:
    if not isinstance(id, str):
        return id

    match id.lower():
        case "variational_elbo":
            return VariationalELBO(likelihood, model, num_data=num_data)
        case "predictive_log":
            return PredictiveLogLikelihood(likelihood, model, num_data=num_data)
        case "variational_marginal_log":
            return VariationalMarginalLogLikelihood(likelihood, model, num_data=num_data)
        case "gamma_robust_variational_elbo":
            return GammaRobustVariationalELBO(likelihood, model, num_data=num_data)
        case _:
            raise ValueError(f"invalid objective function id: {id}")
