from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module, Parameter, init


BasisFunction = Callable[[Tensor], Tensor]


def get_basis_function(id: str) -> BasisFunction | None:
    match id.lower():
        case "gaussian":
            return gaussian
        case "linear":
            return linear
        case "quadratic":
            return quadratic
        case "inverse_quadratic":
            return inverse_quadratic
        case "multiquadric":
            return multiquadric
        case "inverse_multiquadric":
            return inverse_multiquadric
        case "spline":
            return spline
        case "poisson_one":
            return poisson_one
        case "poisson_two":
            return poisson_two
        case "matern32":
            return matern32
        case "matern52":
            return matern52

    return None


def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi


def linear(alpha):
    phi = alpha
    return phi


def quadratic(alpha):
    phi = alpha.pow(2)
    return phi


def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi


def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi


def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi


def poisson_two(alpha):
    phi = ((alpha - 2*torch.ones_like(alpha)) / 2*torch.ones_like(alpha)) \
        * alpha * torch.exp(-alpha)
    return phi


def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
    return phi


def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5**0.5*alpha + (5/3)
           * alpha.pow(2))*torch.exp(-5**0.5*alpha)
    return phi


class RBFRegressor(Module):
    def __init__(self, *,
                 in_features: int,
                 out_features: int,
                 basis_func: str = "gaussian",
                 ) -> None:
        super(RBFRegressor, self).__init__()

        self.in_features: int = in_features
        self.out_features: int = out_features

        self.centres = Parameter(Tensor(out_features, in_features))
        self.log_sigmas = Parameter(Tensor(out_features))

        basis_function: BasisFunction | None = get_basis_function(basis_func)
        if basis_function is None:
            raise ValueError(f"Unknown basis function: {basis_func}")

        self.basis_function = basis_function

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.centres, 0, 1)
        init.constant_(self.log_sigmas, 0)

    def forward(self, input: Tensor) -> Tensor:
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)

        return self.basis_function(distances).squeeze(-1)
