import torch
import numpy as np
from typing import NamedTuple

from torch import Tensor
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Bernoulli

from gpytorch.optim import NGD
from gpytorch.mlls import VariationalELBO
from gpytorch.distributions.multivariate_normal import MultivariateNormal

from vclog import Logger

# TMP
from ..models.gp_binary_classifier import GPModel, PGLikelihood


class GPBCInfo(NamedTuple):
    decision: np.ndarray
    mean: np.ndarray
    deviation: np.ndarray
    variance: np.ndarray


class GPBinaryClassifier:
    def __init__(self, inducing_points: int, n_dimensions: int) -> None:
        # Variables
        self.inducing_points: int = inducing_points
        self.n_dimensions: int = n_dimensions
        self._logger: Logger = Logger('gpbc')
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype: torch.dtype | None = torch.float32

        # Models
        inducing_points_tensor: Tensor = torch.tensor(np.random.uniform(-1.0, 1.0, (inducing_points, n_dimensions)), device=self.device, dtype=self.dtype)
        self.model: GPModel = GPModel(inducing_points_tensor, n_dimensions).to(self.device)
        self.likelihood: PGLikelihood = PGLikelihood().to(self.device)

    def train(self,
              input_data: np.ndarray,
              labels: np.ndarray,
              epochs: int = 10,
              batch_size: int = 256,
              variational_lr: float = 0.5,
              hyperparameter_lr: float = 0.1,
              ) -> None:
        # Function requirements
        if not np.all(np.logical_and(input_data >= -1, input_data <= 1)):
            raise ValueError('input data must be in the interval [-1, 1]')

        if not np.all(np.logical_or(labels == 0, labels == 1)):
            raise ValueError('labels must be either 0 or 1')

        self.model.train()
        self.likelihood.train()

        # Dataset initialization
        train_x: Tensor = torch.tensor(input_data, device=self.device, dtype=self.dtype)
        train_y: Tensor = torch.tensor(labels, device=self.device, dtype=self.dtype)

        dataset: TensorDataset = TensorDataset(train_x, train_y)
        data_loader: DataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizers
        variational_ngd_optimizer: NGD = NGD(self.model.variational_parameters(), num_data=train_y.size(0), lr=variational_lr)
        hyperparameter_optimizer: Adam = Adam([{'params': self.model.hyperparameters()},
                                               {'params': self.likelihood.parameters()},
                                               ], lr=hyperparameter_lr)

        loss_function: VariationalELBO = VariationalELBO(self.likelihood, self.model, num_data=train_y.size(0))

        for i in range(epochs):
            for j, (x_batch, y_batch) in enumerate(data_loader):
                variational_ngd_optimizer.zero_grad()
                hyperparameter_optimizer.zero_grad()

                output: MultivariateNormal = self.model(x_batch)

                loss: Tensor = -loss_function(output, y_batch)  # type: ignore
                loss.backward()

                variational_ngd_optimizer.step()
                hyperparameter_optimizer.step()

                self._logger.info(f'Training in progress... {int((i/epochs)*100+(j/len(data_loader))*100/epochs)}%', flush=True)

        self._logger.info('Training in progress... 100%')

    @torch.no_grad()
    def predict(self, input_data: np.ndarray, beta: float = 0.5) -> GPBCInfo:
        if not input_data.size - len(input_data):
            input_data = np.array([input_data])

        # Initialize tensors
        test_x: Tensor = torch.tensor(input_data, device=self.device, dtype=self.dtype)

        self.model.eval()
        self.likelihood.eval()

        predictions: Bernoulli = self.likelihood(self.model(test_x))  # type:ignore
        mean: np.ndarray = predictions.mean.cpu().numpy()  # type:ignore
        deviation: np.ndarray = predictions.stddev.cpu().numpy()  # type: ignore
        variance: np.ndarray = predictions.variance.cpu().numpy()  # type: ignore

        decision: np.ndarray = mean + beta*deviation

        return GPBCInfo(decision=decision,
                        mean=mean,
                        deviation=deviation,
                        variance=variance,
                        )

    def save(self, root: str) -> None:
        torch.save(self.model.state_dict(), root+'model.pth')
        torch.save(self.likelihood.state_dict(), root+'likelihood.pth')
        np.savetxt(root+'class_data.txt', [self.inducing_points, self.n_dimensions], delimiter=',')

    @classmethod
    def from_model(cls, root: str):
        inducing_points, n_dimensions = np.loadtxt(root+'class_data.txt')
        gpbc = cls(int(inducing_points), int(n_dimensions))
        gpbc.model.load_state_dict(torch.load(root+'model.pth', map_location=torch.device('cpu')))
        gpbc.likelihood.load_state_dict(torch.load(root+'likelihood.pth', map_location=torch.device('cpu')))

        return gpbc
