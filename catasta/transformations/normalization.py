import numpy as np

from .transformation import Transformation


class Normalization(Transformation):
    def __init__(self, norm_tech: str) -> None:
        """
        Initialize the normalization technique

        Parameters
        ----------
        norm_tech : str
            Normalization technique to be applied. The following techniques are supported:
            - "arctan"
            - "boxcox"
            - "decimal"
            - "logistic"
            - "mad"
            - "max"
            - "maxabs"
            - "maxmin"
            - "mean"
            - "median"
            - "min"
            - "minabs"
            - "minmax"
            - "power"
            - "quantile"
            - "robust"
            - "softmax"
            - "std"
            - "tanh"
            - "var"
            - "yeojohnson"
            - "zscore"
        """
        self.norm_tech: str = norm_tech

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply normalization technique to the input data

        Parameters
        ----------
        x : np.ndarray
            Input data to be normalized

        Returns
        -------
        np.ndarray
            Normalized data
        """
        match self.norm_tech:
            case "arctan":
                return np.arctan(x)
            case "boxcox":
                return np.sign(x) * np.power(np.abs(x), 1 / 2)
            case "decimal":
                return x / 10 ** np.ceil(np.log10(np.max(np.abs(x))))
            case "logistic":
                return 1 / (1 + np.exp(-x))
            case "mad":
                return x / np.median(np.abs(x - np.median(x)))
            case "max":
                return x / np.max(x)
            case "maxabs":
                return x / np.max(np.abs(x))
            case "maxmin":
                return x / (np.max(x) - np.min(x))
            case "mean":
                return x / np.mean(x)
            case "median":
                return x / np.median(x)
            case "min":
                return x / np.min(x)
            case "minabs":
                return x / np.min(np.abs(x))
            case "minmax":
                return (x - np.min(x)) / (np.max(x) - np.min(x))
            case "power":
                return np.sign(x) * np.power(np.abs(x), 1 / 2)
            case "quantile":
                return x / np.quantile(x, 0.95)
            case "robust":
                return x / (np.percentile(x, 75) - np.percentile(x, 25))
            case "softmax":
                return np.exp(x) / np.sum(np.exp(x))
            case "std":
                return x / np.std(x)
            case "tanh":
                return np.tanh(x)
            case "var":
                return x / np.var(x)
            case "yeojohnson":
                return np.sign(x) * np.power(np.abs(x), 1 / 2)
            case "zscore":
                return (x - np.mean(x)) / np.std(x)
            case _:
                raise ValueError("Unknown normalization technique")
