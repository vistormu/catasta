import numpy as np
import matplotlib.pyplot as plt

from catasta.models import TransformerRegressor, ApproximateGPRegressor, FFTTransformerRegressor
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.entities import RegressionEvalInfo, RegressionTrainInfo, RegressionPrediction

from vclog import Logger

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from torch.distributions import Distribution

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal

from gpytorch.mlls import (
    MarginalLogLikelihood,
    VariationalELBO,
    PredictiveLogLikelihood,
    VariationalMarginalLogLikelihood,
)

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

    return None


def get_objective_function(id: str, model: Module, likelihood: Module, num_data: int) -> MarginalLogLikelihood | None:
    match id.lower():
        case "variational_elbo":
            return VariationalELBO(likelihood, model, num_data=num_data)
        case "predictive_log":
            return PredictiveLogLikelihood(likelihood, model, num_data=num_data)
        case "variational_marginal_log":
            return VariationalMarginalLogLikelihood(likelihood, model, num_data=num_data)


class GaussianRegressionScaffold:
    def __init__(self, *,
                 transformer_model: Module,
                 model: Module,
                 dataset: RegressionDataset,
                 optimizer: str = "adam",
                 loss_function: str = "variational_elbo",
                 ) -> None:
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype: torch.dtype = torch.float32

        self.transformer_model: Module = transformer_model.to(self.device)

        self.model: Module = model.to(self.device)
        self.likelihood = GaussianLikelihood().to(self.device)
        self.dataset: RegressionDataset = dataset

        self.optimizer_id: str = optimizer
        self.loss_function_id: str = loss_function

        self.logger: Logger = Logger("catasta")
        self.logger.info(f"using device: {self.device}")

    def train(self,
              epochs: int = 100,
              batch_size: int = 32,
              lr: float = 1e-3,
              ) -> RegressionTrainInfo:
        self.transformer_model.eval()
        self.model.train()
        self.likelihood.train()

        data_loader: DataLoader = DataLoader(self.dataset.train, batch_size=batch_size, shuffle=False)

        optimizer: Optimizer | None = get_optimizer(self.optimizer_id, [self.model, self.likelihood], lr)
        if optimizer is None:
            raise ValueError(f"invalid optimizer id: {self.optimizer_id}")

        mll: MarginalLogLikelihood | None = get_objective_function(self.loss_function_id, self.model, self.likelihood, len(self.dataset.train))
        if mll is None:
            raise ValueError(f"invalid loss function id: {self.loss_function_id}")

        losses: list[float] = []
        for i in range(epochs):
            batch_losses: list[float] = []
            for j, (x_batch, y_batch) in enumerate(data_loader):
                optimizer.zero_grad()

                x_batch = x_batch.to(self.device, dtype=self.dtype)
                y_batch = y_batch.to(self.device, dtype=self.dtype)

                x_batch = self.transformer_model.encode(x_batch)

                output: MultivariateNormal = self.model(x_batch)

                loss: Tensor = -mll(output, y_batch)  # type: ignore
                batch_losses.append(loss.item())
                loss.backward()

                optimizer.step()

                self.logger.info(f"epoch {i}/{epochs} | {int((i/epochs)*100+(j/len(data_loader))*100/epochs)}% | loss: {loss.item():.4f}", flush=True)

            losses.append(np.mean(batch_losses).astype(float))

        self.logger.info(f'epoch {epochs}/{epochs} | 100% | loss: {np.min(losses):.4f}')

        return RegressionTrainInfo(np.array(losses))

    @torch.no_grad()
    def predict(self, input: np.ndarray | Tensor) -> RegressionPrediction:
        self.model.eval()
        self.likelihood.eval()
        self.transformer_model.eval()

        input_tensor: Tensor = torch.tensor(input) if isinstance(input, np.ndarray) else input
        input_tensor = input_tensor.to(self.device, dtype=self.dtype)

        input_tensor = self.transformer_model.encode(input_tensor)

        output: Distribution = self.likelihood(self.model(input_tensor))

        mean: np.ndarray = output.mean.cpu().numpy()
        std: np.ndarray = output.stddev.cpu().numpy()

        return RegressionPrediction(mean, std)

    @torch.no_grad()
    def evaluate(self) -> RegressionEvalInfo:
        test_index: int = np.floor(len(self.dataset) * self.dataset.splits[0]).astype(int)
        test_x: np.ndarray = self.dataset.inputs[test_index:]
        test_y: np.ndarray = self.dataset.outputs[test_index:].flatten()

        if self.dataset.test is None:
            raise ValueError(f"test split must be greater than 0")

        data_loader = DataLoader(self.dataset.test, batch_size=1, shuffle=False)

        means: np.ndarray = np.array([])
        stds: np.ndarray = np.array([])
        for x_batch, _ in data_loader:
            output: RegressionPrediction = self.predict(x_batch)
            means = np.concatenate([means, output.prediction])
            if output.stds is not None:
                stds = np.concatenate([stds, output.stds])

        if len(means) != len(test_y):
            min_len: int = min(len(means), len(test_y))
            means = means[-min_len:]
            stds = stds[-min_len:]
            test_y = test_y[-min_len:]

        return RegressionEvalInfo(test_x, test_y, means, stds)


def main() -> None:
    # dataset_root: str = "tests/data/wire_lisbeth/strain/"
    dataset_root: str = "tests/data/nylon_carmen/strain/"

    context_length: int = 1024
    d_model: int = 24

    dataset = RegressionDataset(
        root=dataset_root,
        context_length=context_length,
        splits=(6/7, 1/7, 0.0),
    )
    model = FFTTransformerRegressor(
        context_length=context_length,
        n_patches=32,
        output_dim=1,
        d_model=d_model,
        n_heads=4,
        n_layers=4,
        feedforward_dim=32,
        head_dim=8,
    )
    scaffold = RegressionScaffold(
        model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="huber",
    )

    train_info: RegressionTrainInfo = scaffold.train(
        epochs=100,
        batch_size=64,
        lr=1e-3,
    )

    plt.figure(figsize=(30, 20))
    plt.plot(train_info.train_loss, label="train loss", color="black")
    plt.plot(train_info.eval_loss, label="eval loss", color="red")  # type: ignore
    plt.legend()
    plt.show()

    Logger.debug(f"min train loss: {np.min(train_info.train_loss):.4f}, "
                 f"min eval loss: {np.min(train_info.eval_loss):.4f}")  # type: ignore

    info: RegressionEvalInfo = scaffold.evaluate()

    plt.figure(figsize=(30, 20))
    plt.plot(info.real, label="real", color="black")
    plt.plot(info.predicted, label="predictions", color="red")
    plt.legend()
    plt.show()

    Logger.debug(info)

    # GP MODEL
    n_inducing_points: int = 128
    dataset = RegressionDataset(
        root=dataset_root,
        context_length=context_length,
        prediction_length=1,
        splits=(6/7, 0.0, 1/7),
    )
    gp_model = ApproximateGPRegressor(
        n_inducing_points=n_inducing_points,
        n_inputs=d_model,
        kernel="rq",
        mean="constant"
    )
    scaffold = GaussianRegressionScaffold(
        model=gp_model,
        transformer_model=model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="variational_elbo",
    )

    train_info: RegressionTrainInfo = scaffold.train(
        epochs=100,
        batch_size=64,
        lr=1e-3,
    )

    Logger.debug(f"min train loss: {np.min(train_info.train_loss):.4f}")

    plt.figure(figsize=(30, 20))
    plt.plot(train_info.train_loss, label="train loss", color="black")
    plt.legend()
    plt.show()

    info: RegressionEvalInfo = scaffold.evaluate()

    plt.figure(figsize=(30, 20))
    plt.plot(info.predicted, label="predictions", color="red")
    plt.plot(info.real, label="real", color="black")
    plt.fill_between(range(len(info.predicted)), info.predicted-1*info.stds, info.predicted+1*info.stds, color="red", alpha=0.2)  # type: ignore
    plt.legend()
    plt.show()

    Logger.debug(info)


if __name__ == '__main__':
    main()
