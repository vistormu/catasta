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


def get_loss_function(id: str) -> _Loss:
    loss_function: _Loss | None = loss_functions.get(id.lower(), None)
    if loss_function is None:
        raise ValueError(f"Loss function {id} not found")

    return loss_function
