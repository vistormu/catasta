import numpy as np
import matplotlib.pyplot as plt

from catasta.models import ApproximateGPRegressor
from catasta.datasets import RegressionDataset
from catasta.dataclasses import RegressionEvalInfo, RegressionTrainInfo, RegressionPrediction
from catasta.utils import get_optimizer, get_objective_function

from vclog import Logger

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.distributions import Distribution

from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from torch import nn
from torch.fft import fft
from torch.nn import Module, Sequential, LayerNorm, Linear, GELU, Softmax

from einops import rearrange
from einops.layers.torch import Rearrange

from torch.nn import Module, Linear, Sequential, ReLU, BatchNorm1d, Dropout


class FeedforwardRegressor(Module):
    def __init__(self, *,
                 input_dim: int,
                 hidden_dims: list[int],
                 output_dim: int,
                 dropout: float,
                 ) -> None:
        super().__init__()

        n_layers: int = len(hidden_dims)
        layers: list[Module] = []
        for i in range(n_layers):
            if i == 0:
                layers.append(Linear(input_dim, hidden_dims[i]))
            else:
                layers.append(Linear(hidden_dims[i-1], hidden_dims[i]))

            layers.append(BatchNorm1d(hidden_dims[i]))
            layers.append(ReLU())
            layers.append(Dropout(dropout))

        layers.append(Linear(hidden_dims[-1], output_dim))

        self.model: Sequential = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)

        return x


def posemb_sincos_1d(patches: Tensor, temperature: int = 10000) -> Tensor:
    n: int = patches.shape[1]
    d_model: int = patches.shape[2]
    device: torch.device = patches.device
    dtype: torch.dtype = patches.dtype

    if (d_model % 2) != 0:
        raise ValueError(f'feature dimension must be multiple of 2 for sincos emb. got {d_model}')

    n_tensor: Tensor = torch.arange(n, device=device)
    omega: Tensor = torch.arange(d_model // 2, device=device) / (d_model // 2 - 1)
    omega = 1.0 / (temperature ** omega)

    n_tensor = n_tensor.flatten()[:, None] * omega[None, :]
    pe: Tensor = torch.cat((n_tensor.sin(), n_tensor.cos()), dim=1)

    return pe.to(dtype)


class FeedForward(Module):
    def __init__(self, *,
                 d_model: int,
                 hidden_dim: int,
                 ) -> None:
        super().__init__()

        self.net = Sequential(
            LayerNorm(d_model),
            Linear(d_model, hidden_dim),
            GELU(),
            Linear(hidden_dim, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, *,
                 d_model: int,
                 n_heads: int = 8,
                 head_dim: int = 64,
                 ) -> None:
        super().__init__()

        inner_dim: int = head_dim * n_heads
        self.n_heads: int = n_heads
        self.scale: int = head_dim ** -0.5
        self.norm = LayerNorm(d_model)

        self.attend = Softmax(dim=-1)

        self.to_qkv = Linear(d_model, inner_dim * 3, bias=False)
        self.to_out = Linear(inner_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)

        qkv: Tensor = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.n_heads), qkv)

        dots: Tensor = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn: Tensor = self.attend(dots)

        out: Tensor = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, *,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 head_dim: int,
                 feedforward_dim: int,
                 ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                Attention(d_model=d_model, n_heads=n_heads, head_dim=head_dim),
                FeedForward(d_model=d_model, hidden_dim=feedforward_dim)
            ]))

    def forward(self, x: Tensor) -> Tensor:
        for attn, ff in self.layers:  # type: ignore
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class FFTTransformerRegressor(Module):
    def __init__(self, *,
                 context_length: int,
                 n_patches: int,
                 output_dim: int,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 feedforward_dim: int,
                 head_dim: int,
                 ) -> None:
        super().__init__()
        patch_dim: int = context_length // n_patches
        freq_patch_dim: int = patch_dim * 2

        if context_length % patch_dim != 0:
            raise ValueError(f"sequence length {context_length} must be divisible by patch size {patch_dim}")

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_dim),
            LayerNorm(patch_dim),
            Linear(patch_dim, d_model),
            LayerNorm(d_model),
        )

        self.to_freq_embedding = nn.Sequential(
            Rearrange('b c (n p) ri -> b n (p ri c)', p=patch_dim),
            LayerNorm(freq_patch_dim),
            Linear(freq_patch_dim, d_model),
            LayerNorm(d_model),
        )

        self.transformer = Transformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            feedforward_dim=feedforward_dim,
        )

        self.linear_head = nn.Linear(d_model, output_dim)

    def forward(self, input: Tensor) -> Tensor:
        input = rearrange(input, 'b s -> b 1 s')
        freqs: Tensor = torch.view_as_real(fft(input))

        x: Tensor = self.to_patch_embedding(input)
        f: Tensor = self.to_freq_embedding(freqs)

        x += posemb_sincos_1d(x)
        f += posemb_sincos_1d(f)

        x = torch.cat((x, f), dim=1)
        x = self.transformer(x)
        x = x.mean(dim=1)

        return x


class GaussianRegressionScaffold:
    def __init__(self, *,
                 pre_model: Module,
                 model: Module,
                 dataset: RegressionDataset,
                 optimizer: str = "adam",
                 loss_function: str = "variational_elbo",
                 ) -> None:
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype: torch.dtype = torch.float32

        self.pre_model: Module = pre_model.to(self.device)

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
        self.pre_model.train()
        self.model.train()
        self.likelihood.train()

        data_loader: DataLoader = DataLoader(self.dataset.train, batch_size=batch_size, shuffle=False)

        optimizer: Optimizer | None = get_optimizer(self.optimizer_id, [self.model, self.likelihood, self.pre_model], lr)
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

                x_batch = self.pre_model(x_batch)
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
        self.pre_model.eval()

        input_tensor: Tensor = torch.tensor(input) if isinstance(input, np.ndarray) else input
        input_tensor = input_tensor.to(self.device, dtype=self.dtype)

        input_tensor = self.pre_model(input_tensor)

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
    dataset_root: str = "tests/data/nylon_carmen/paper/strain/mixed_10_20/"
    # dataset_root: str = "tests/data/nylon_carmen_elasticband/paper/strain/mixed_10_20/"

    context_length: int = 1024
    d_model: int = 32

    # 512, 4, 1, 16, 2, 2, 32, 4
    # dataset = RegressionDataset(
    #     root=dataset_root,
    #     context_length=context_length,
    #     splits=(7/8, 1/8, 0.0),
    # )
    pre_model = FFTTransformerRegressor(
        context_length=context_length,
        n_patches=4,
        output_dim=1,
        d_model=d_model,
        n_heads=2,
        n_layers=2,
        feedforward_dim=32,
        head_dim=16,
    )
    # pre_model = FeedforwardRegressor(
    #     input_dim=context_length,
    #     hidden_dims=[128, 64, 32],
    #     output_dim=d_model,
    #     dropout=0.2,
    # )
    n_inducing_points: int = 128
    dataset = RegressionDataset(
        root=dataset_root,
        context_length=context_length,
        prediction_length=1,
        splits=(7/8, 0.0, 1/8),
    )
    gp_model = ApproximateGPRegressor(
        n_inducing_points=n_inducing_points,
        n_inputs=d_model,
        kernel="rff",
        mean="zero"
    )
    scaffold = GaussianRegressionScaffold(
        model=gp_model,
        pre_model=pre_model,
        dataset=dataset,
        optimizer="adamw",
        loss_function="variational_marginal_log",
    )

    train_info: RegressionTrainInfo = scaffold.train(
        epochs=1000,
        batch_size=128,
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
