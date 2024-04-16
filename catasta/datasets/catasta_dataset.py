from torch.utils.data import Dataset


class CatastaDataset(Dataset):
    def __init__(self) -> None:
        self.train: Dataset
        self.validation: Dataset
        self.test: Dataset
