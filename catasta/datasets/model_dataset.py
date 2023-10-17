import os

import pandas as pd

from ..entities import ModelData


class ModelDataset:
    def __init__(self, *,
                 root: str,
                 ) -> None:
        self.data: list[ModelData] = []
        for filename in os.listdir(root):
            if not filename.endswith(".csv"):
                continue

            data_frame: pd.DataFrame = pd.read_csv(root + filename)
            self.data.append(ModelData(
                input=data_frame['input'].to_numpy(),
                output=data_frame['output'].to_numpy(),
            ))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> ModelData:
        return self.data[index]
