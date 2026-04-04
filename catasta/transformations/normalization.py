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

    @staticmethod
    def _safe_divide(numerator: np.ndarray, denominator: float, eps: float = 1e-12) -> np.ndarray:
        if abs(float(denominator)) <= eps:
            return np.zeros_like(numerator, dtype=float)
        return numerator / denominator

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
        x = np.asarray(x, dtype=float)

        match self.norm_tech:
            case "arctan":
                return np.arctan(x)
            case "boxcox":
                return np.sign(x) * np.power(np.abs(x), 1 / 2)
            case "decimal":
                max_abs = float(np.max(np.abs(x)))
                if max_abs <= 0.0:
                    return np.zeros_like(x, dtype=float)
                scale = 10 ** np.ceil(np.log10(max_abs))
                return self._safe_divide(x, float(scale))
            case "logistic":
                return 1 / (1 + np.exp(-x))
            case "mad":
                mad = float(np.median(np.abs(x - np.median(x))))
                return self._safe_divide(x, mad)
            case "max":
                return self._safe_divide(x, float(np.max(x)))
            case "maxabs":
                return self._safe_divide(x, float(np.max(np.abs(x))))
            case "maxmin":
                return self._safe_divide(x, float(np.max(x) - np.min(x)))
            case "mean":
                return self._safe_divide(x, float(np.mean(x)))
            case "median":
                return self._safe_divide(x, float(np.median(x)))
            case "min":
                return self._safe_divide(x, float(np.min(x)))
            case "minabs":
                return self._safe_divide(x, float(np.min(np.abs(x))))
            case "minmax":
                min_value = float(np.min(x))
                range_value = float(np.max(x) - min_value)
                return self._safe_divide(x - min_value, range_value)
            case "power":
                return np.sign(x) * np.power(np.abs(x), 1 / 2)
            case "quantile":
                quantile = float(np.quantile(x, 0.95))
                return self._safe_divide(x, quantile)
            case "robust":
                iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
                return self._safe_divide(x, iqr)
            case "softmax":
                shifted = x - np.max(x)
                exp_shifted = np.exp(shifted)
                return self._safe_divide(exp_shifted, float(np.sum(exp_shifted)))
            case "std":
                return self._safe_divide(x, float(np.std(x)))
            case "tanh":
                return np.tanh(x)
            case "var":
                return self._safe_divide(x, float(np.var(x)))
            case "yeojohnson":
                return np.sign(x) * np.power(np.abs(x), 1 / 2)
            case "zscore":
                return self._safe_divide(x - np.mean(x), float(np.std(x)))
            case _:
                raise ValueError("Unknown normalization technique")
