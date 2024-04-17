from torch.nn import Module, Identity

from gpytorch.likelihoods import (
    GaussianLikelihood,
    BernoulliLikelihood,
    LaplaceLikelihood,
    SoftmaxLikelihood,
    StudentTLikelihood,
    BetaLikelihood,
)


id_to_likelihood = {
    "gaussian": GaussianLikelihood,
    "bernoulli": BernoulliLikelihood,
    "laplace": LaplaceLikelihood,
    "softmax": SoftmaxLikelihood,
    "studentt": StudentTLikelihood,
    "beta": BetaLikelihood,
}


def get_likelihood(id: str | Module | None) -> Module:
    if id is None:
        return Identity()
    if isinstance(id, Module):
        return id

    likelihood = id_to_likelihood.get(id.lower(), None)
    if likelihood is None:
        raise ValueError(f"Likelihood {id} not recognized")

    return likelihood()
