from torch.optim import Adam, SGD, AdamW, Optimizer

from ....models import Regressor


def get_optimizer(id: str, model: Regressor, lr: float) -> Optimizer | None:
    match id.lower():
        case "adam":
            return Adam(model.parameters(), lr=lr)
        case "sgd":
            return SGD(model.parameters(), lr=lr)
        case "adamw":
            return AdamW(model.parameters(), lr=lr)

    return None
