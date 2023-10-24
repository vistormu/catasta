import numpy as np


class EvalInfo:
    def __init__(self, predicted: np.ndarray, real: np.ndarray) -> None:
        self.mae: float = np.mean(np.abs(predicted - real))
        self.mse: float = np.mean((predicted - real)**2)
        self.rmse: float = np.sqrt(np.mean((predicted - real)**2))
        self.r2: float = 1 - np.sum((predicted - real)**2) / np.sum((real - np.mean(real))**2).astype(float)
        self.mape: float = np.mean(np.abs((predicted - real) / real))
        self.smape: float = np.mean(np.abs(predicted - real) / (np.abs(predicted) + np.abs(real)))
        self.mase: float = np.mean(np.abs(predicted - real) / np.mean(np.abs(real)))
        self.masep: float = np.mean(np.abs(predicted - real) / np.mean(np.abs(real - np.mean(real))))

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
