import torch
from torch import Tensor
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageDataset(Dataset):
    def __init__(self, *,
                 root: str,
                 ) -> None:
        self.dataset: ImageFolder = ImageFolder(root, transform=transforms.ToTensor())
        self.image_size: int = self.dataset[0][0].shape[1]
        self.n_classes: int = len(self.dataset.classes)
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image, label = self.dataset[index]

        return image.to(self.device), torch.tensor(label).to(self.device)
