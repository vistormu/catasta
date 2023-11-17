from torch.nn.modules.loss import _Loss
from torch.nn import (
    MSELoss,
    L1Loss,
    SmoothL1Loss,
    HuberLoss,
)


def get_loss_function(id: str) -> _Loss | None:
    match id.lower():
        case "mse":
            return MSELoss()
        case "l1":
            return L1Loss()
        case "smooth_l1":
            return SmoothL1Loss()
        case "huber":
            return HuberLoss()

    return None
