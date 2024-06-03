import torch

torch.backends.cudnn.benchmark = True


def set_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
