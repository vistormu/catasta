import numpy as np


class RegressionEvalInfo:
    def __init__(self, true_input: np.ndarray, true_output: np.ndarray, predicted_output: np.ndarray, predicted_std: np.ndarray | None = None) -> None:
        self.true_input: np.ndarray = true_input
        self.predicted_output: np.ndarray = predicted_output
        self.true_output: np.ndarray = true_output
        self.predicted_std: np.ndarray | None = predicted_std

        self.mae: float = np.mean(np.abs(predicted_output - true_output)).astype(float)
        self.mse: float = np.mean((predicted_output - true_output)**2).astype(float)
        self.rmse: float = np.sqrt(np.mean((predicted_output - true_output)**2))
        self.r2: float = 1 - np.sum((predicted_output - true_output)**2) / np.sum((true_output - np.mean(true_output))**2).astype(float)
        self.mape: float = np.mean(np.abs((predicted_output - true_output) / true_output)).astype(float)
        self.smape: float = np.mean(np.abs(predicted_output - true_output) / (np.abs(predicted_output) + np.abs(true_output))).astype(float)
        self.mase: float = np.mean(np.abs(predicted_output - true_output) / np.mean(np.abs(true_output))).astype(float)
        self.masep: float = np.mean(np.abs(predicted_output - true_output) / np.mean(np.abs(true_output - np.mean(true_output)))).astype(float)

    def __repr__(self) -> str:
        mae_str: str = f"MAE: {self.mae:.4f}\n"
        mse_str: str = f"MSE: {self.mse:.4f}\n"
        rmse_str: str = f"RMSE: {self.rmse:.4f}\n"
        r2_str: str = f"R2: {self.r2:.4f}\n"
        mape_str: str = f"MAPE: {self.mape:.4f}\n"
        smape_str: str = f"SMAPE: {self.smape:.4f}\n"
        mase_str: str = f"MASE: {self.mase:.4f}\n"
        masep_str: str = f"MASEP: {self.masep:.4f}\n"

        return mae_str + mse_str + rmse_str + r2_str + mape_str + smape_str + mase_str + masep_str
