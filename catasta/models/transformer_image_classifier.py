from torch import Tensor
from torch.nn import Module
from torch.nn.functional import cross_entropy, sigmoid
from vit_pytorch import SimpleViT


class TransformerImageClassifier(Module):
    def __init__(self, *,
                 image_size: int,
                 patch_size: int,
                 embedding_dim: int,
                 n_heads: int,
                 n_layers: int,
                 feedforward_dim: int,
                 n_classes: int,
                 ) -> None:
        super().__init__()

        self.vit = SimpleViT(image_size=image_size,
                             patch_size=patch_size,
                             num_classes=n_classes,
                             dim=embedding_dim,
                             depth=n_layers,
                             heads=n_heads,
                             mlp_dim=feedforward_dim,
                             channels=3,
                             )

    def forward(self, x: Tensor, target: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        x = self.vit(x)

        loss: Tensor | None = None
        if target is not None:
            loss = cross_entropy(input=x, target=target)
        else:
            x = sigmoid(x)

        return x, loss
