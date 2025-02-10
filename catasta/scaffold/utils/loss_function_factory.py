from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.nn import (
    MSELoss,
    L1Loss,
    SmoothL1Loss,
    HuberLoss,
    PoissonNLLLoss,
    KLDivLoss,
    CrossEntropyLoss,
    NLLLoss,
    BCELoss,
    BCEWithLogitsLoss,
    MarginRankingLoss,
    HingeEmbeddingLoss,
    MultiLabelMarginLoss,
    MultiLabelSoftMarginLoss,
)

from gpytorch.mlls import (
    VariationalELBO,
    PredictiveLogLikelihood,
    VariationalMarginalLogLikelihood,
    GammaRobustVariationalELBO,
)
from gpytorch.models.gp import GP

loss_functions: dict[str, _Loss] = {
    "mse": MSELoss(),
    "l1": L1Loss(),
    "smooth_l1": SmoothL1Loss(),
    "huber": HuberLoss(),
    "poisson": PoissonNLLLoss(),
    "kl_div": KLDivLoss(),
    "cross_entropy": CrossEntropyLoss(),
    "nll": NLLLoss(),
    "bce": BCELoss(),
    "bce_with_logits": BCEWithLogitsLoss(),
    "margin_ranking": MarginRankingLoss(),
    "hinge_embedding": HingeEmbeddingLoss(),
    "multi_label_margin": MultiLabelMarginLoss(),
    "multi_label_soft_margin": MultiLabelSoftMarginLoss(),
}


def get_loss_function(id: str | Module, model: Module, num_data: int) -> Module:
    if not isinstance(id, str):
        return id

    loss_function: _Loss | None = loss_functions.get(id.lower(), None)
    if loss_function is not None and isinstance(model, GP):
        raise ValueError(f"loss function '{id}' is not compatible with GP models")

    if loss_function is None:
        match id.lower():
            case "variational_elbo":
                return VariationalELBO(model.likelihood, model, num_data=num_data)
            case "predictive_log":
                return PredictiveLogLikelihood(model.likelihood, model, num_data=num_data)
            case "variational_marginal_log":
                return VariationalMarginalLogLikelihood(model.likelihood, model, num_data=num_data)
            case "gamma_robust_variational_elbo":
                return GammaRobustVariationalELBO(model.likelihood, model, num_data=num_data)
            case _:
                raise ValueError(f"invalid objective function id: {id}")

    return loss_function
