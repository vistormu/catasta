from torch.nn import Module
from torch.optim import (
    Optimizer,
    Adam,
    SGD,
    AdamW,
    LBFGS,
    RMSprop,
    Rprop,
    Adadelta,
    Adagrad,
    Adamax,
    ASGD,
    SparseAdam,
)


def get_optimizer(id: str, model: Module | list[Module], lr: float) -> Optimizer:
    if isinstance(model, Module):
        model = [model]

    parameters = []
    for m in model:
        parameters += list(m.parameters())

    match id.lower():
        case "adam":
            return Adam(parameters, lr=lr)
        case "sgd":
            return SGD(parameters, lr=lr)
        case "adamw":
            return AdamW(parameters, lr=lr)
        case "lbfgs":
            return LBFGS(parameters, lr=lr)
        case "rmsprop":
            return RMSprop(parameters, lr=lr)
        case "rprop":
            return Rprop(parameters, lr=lr)
        case "adadelta":
            return Adadelta(parameters, lr=lr)
        case "adagrad":
            return Adagrad(parameters, lr=lr)
        case "adamax":
            return Adamax(parameters, lr=lr)
        case "asgd":
            return ASGD(parameters, lr=lr)
        case "sparseadam":
            return SparseAdam(parameters, lr=lr)
        case _:
            raise ValueError(f"invalid optimizer id: {id}")
