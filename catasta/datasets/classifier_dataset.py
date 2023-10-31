import os

from torch import Tensor
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.datasets import ImageFolder


def _check_files(directory: str) -> tuple[bool, bool]:
    has_csv = False
    has_images = False
    for _, _, files in os.walk(directory):
        if any(f.endswith('.csv') for f in files):
            has_csv = True
        if any(f.endswith(('.png', '.jpg', '.jpeg')) for f in files):
            has_images = True
        if has_csv and has_images:
            break
    return has_csv, has_images


class ClassifierDataset(Dataset):
    def __init__(self, *,
                 root: str,
                 ) -> None:
        has_csv, has_img = _check_files(root)

        if not has_csv and not has_img:
            raise FileNotFoundError(f"Neither csv nor image files found in {root}")

        if has_csv and has_img:
            raise FileNotFoundError(f"Both csv and image files found in {root}")

        if has_img:
            self.dataset: ImageFolder = ImageFolder(root, transform=transforms.ToTensor())
            self.image_size: int = self.dataset[0][0].shape[1]
            self.n_classes: int = len(self.dataset.classes)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.dataset[idx]
