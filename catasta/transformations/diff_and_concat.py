import numpy as np

from .transformation import Transformation
from .window_sliding import WindowSliding


class DiffAndConcat(Transformation):
    def __init__(self, *,
                 n_diffs: int,
                 elements_per_diff: int,
                 filter: Transformation,
                 ) -> None:
        self.n_diffs: int = n_diffs
        self.elements_per_diff: int = elements_per_diff
        self.filter: Transformation = filter
        self.window_sliding: WindowSliding = WindowSliding(
            window_size=elements_per_diff,
            stride=1,
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        diffs: np.ndarray = np.zeros((self.n_diffs+1, len(x)))
        diffs[0] = x
        for i in range(1, self.n_diffs+1):
            diffs[i] = np.gradient(self.filter(diffs[i - 1]))

        x = np.concatenate([self.window_sliding(diff) for diff in diffs], axis=1)

        return x.flatten()
