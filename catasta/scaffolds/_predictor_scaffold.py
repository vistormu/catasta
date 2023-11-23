from pathlib import Path

import numpy as np
import pandas as pd

from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.model.forecast import Forecast
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.mx.model.estimator import GluonEstimator
from gluonts.mx.trainer import Trainer
from gluonts.model.predictor import Predictor as GluonPredictor
from gluonts.torch.model.predictor import PyTorchPredictor

from vclog import Logger

from ..dataclasses import ModelData


class PredictorScaffold:
    def __init__(self, model: PyTorchLightningEstimator | GluonEstimator | None = None, saved_model: str | None = None) -> None:
        self.logger = Logger("catasta")

        if model is None and saved_model is None:
            raise ValueError("Must specify a model or a saved model path")
        elif model is not None and saved_model is not None:
            raise ValueError("Must specify only a model or a saved model path")
        elif model is not None:
            self.estimator: PyTorchLightningEstimator | GluonEstimator = model
        elif saved_model is not None:
            self.predictor: GluonPredictor | PyTorchPredictor = self.load(saved_model)
            self.logger.info(f"Loaded model from {saved_model}")

    def train(self, *,
              dataset: list[ModelData],
              epochs: int = 100,
              lr: float = 0.001,
              train_ratio: float = 0.8,
              ) -> None:
        if isinstance(self.estimator, PyTorchLightningEstimator):
            self.estimator.trainer_kwargs["max_epochs"] = epochs
            self.estimator.lr = lr  # type: ignore
        elif isinstance(self.estimator, GluonEstimator):
            self.estimator.trainer = Trainer(epochs=epochs, learning_rate=lr)

        train_len: int = int(len(dataset) * train_ratio)

        train_ds = ListDataset([
            {"target": data.output, "start": "2020-01-01", "feat_dynamic_real": data.input.reshape(1, -1)}
            for data in dataset[:train_len]
        ], freq="D")

        val_ds = ListDataset([
            {"target": data.output, "start": "2020-01-01", "feat_dynamic_real": data.input.reshape(1, -1)}
            for data in dataset[train_len:]
        ], freq="D")

        self.predictor: GluonPredictor | PyTorchPredictor = self.estimator.train(train_ds, val_ds)

    def predict(self, data: ModelData) -> tuple[list[Forecast], list[pd.Series]]:
        test_ds = ListDataset([
            {"target": data.output, "start": "2020-01-01", "feat_dynamic_real": data.input.reshape(1, -1)}
        ], freq="D")

        # Prediction
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds,
            predictor=self.predictor,
        )

        forecasts = list(forecast_it)
        tss = list(ts_it)

        return forecasts, tss

    def evaluate(self, dataset: list[ModelData]) -> float:
        mase_errors: list[float] = []
        for data in dataset:
            forecasts, tss = self.predict(data)

            evaluator = Evaluator()
            agg_metrics, _ = evaluator(
                iter(tss),
                iter(forecasts),
            )

            mase: float = agg_metrics["MASE"]
            print(f"MASE: {mase}")

            mase_errors.append(mase)

        return np.mean(mase_errors)

    def save(self, path: str) -> None:
        self.predictor.serialize(Path(path))
        self.logger.info(f"Saved model to {path}")

    def load(self, path: str) -> GluonPredictor | PyTorchPredictor:
        with open(Path(path) / "type.txt", "r") as f:
            predictor_type: str = f.read()

        if "pytorch" in predictor_type.lower():
            return PyTorchPredictor.deserialize(Path(path))
        else:
            return GluonPredictor.deserialize(Path(path))
