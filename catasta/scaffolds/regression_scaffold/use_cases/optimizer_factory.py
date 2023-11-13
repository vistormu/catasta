from torch.nn import Module
from torch.optim import Adam, SGD, AdamW, Optimizer


def get_optimizer(id: str, model: Module | list[Module], lr: float) -> Optimizer | None:
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

    return None
