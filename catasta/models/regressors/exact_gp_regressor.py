from gpytorch.models import ExactGP


class ExactGPREgressor(ExactGP):
    def __init__(self, num_tasks: int, num_inputs: int, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood):
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RQKernel(ard_num_dims=num_inputs), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
