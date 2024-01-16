from torch import Tensor
from torch.nn import Module, Linear, Sequential, ReLU, BatchNorm1d, Dropout


class FeedforwardRegressor(Module):
    def __init__(self, *,
                 input_dim: int,
                 hidden_dims: list[int],
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

        layers.append(Linear(hidden_dims[-1], 1))

        self.model: Sequential = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)

        return x.squeeze()
