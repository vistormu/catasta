from typing import Any

import numpy as np

from ..log import ansi


class EvalInfo:
    def __init__(
        self,
        task: str,
        true_output: np.ndarray,
        predicted_output: Any,
        true_input: np.ndarray | None = None,
        n_classes: int | None = None,
    ) -> None:
        self.task = task
        self.true_input = None if true_input is None else np.asarray(true_input)
        self.true_output = np.asarray(true_output)
        self.predicted_output = np.asarray(predicted_output)

        if self.task == "regression":
            self._calculate_regression_metrics(self.true_output, self.predicted_output)
        elif self.task in ("classification", "signal_classification"):
            self._calculate_classification_metrics(self.true_output, self.predicted_output, n_classes)
        else:
            raise ValueError("Invalid task type")

    @staticmethod
    def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        result = np.zeros_like(numerator, dtype=float)
        np.divide(numerator, denominator, out=result, where=denominator != 0)
        return result

    def _calculate_regression_metrics(self, y: np.ndarray, y_pred: np.ndarray) -> None:
        if y.ndim == 1:
            y = y[:, np.newaxis]
        if y_pred.ndim == 1:
            y_pred = y_pred[:, np.newaxis]

        if y.shape != y_pred.shape and y.size == y_pred.size:
            y_pred = y_pred.reshape(y.shape)

        self.mae = np.mean(np.abs(y - y_pred), axis=0)
        self.mse = np.mean((y - y_pred) ** 2, axis=0)
        self.rmse = np.sqrt(self.mse)

        ss_res = np.sum((y - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
        raw_r2 = self._safe_divide(ss_tot - ss_res, ss_tot)
        self.r2 = np.where(ss_tot == 0, 0.0, raw_r2)

        if self.r2.size == 1:
            self.r2 = float(self.r2.item())  # type: ignore[assignment]

    def _calculate_classification_metrics(self, y: np.ndarray, y_pred: np.ndarray, n_classes: int | None) -> None:
        y_true = np.asarray(y).reshape(-1)
        predictions = np.asarray(y_pred)

        if predictions.ndim > 1:
            y_labels = np.argmax(predictions, axis=1)
        elif predictions.ndim == 1:
            y_labels = predictions
        else:
            raise ValueError("predicted_output must be a 1D label array or a 2D logits/probabilities array")

        y_labels = np.asarray(y_labels).reshape(-1)
        if y_true.shape[0] != y_labels.shape[0]:
            raise ValueError("true_output and predicted_output must contain the same number of samples")

        if not np.all(np.equal(y_true, y_true.astype(int))):
            raise ValueError("true labels must be integer class ids")
        if not np.all(np.equal(y_labels, y_labels.astype(int))):
            raise ValueError("predicted labels must be integer class ids")

        y_true = y_true.astype(int)
        y_labels = y_labels.astype(int)

        if n_classes is None:
            n_classes = int(max(np.max(y_true), np.max(y_labels))) + 1 if y_true.size else 0
        if n_classes <= 0:
            raise ValueError("n_classes must be a positive integer")

        self.confusion_matrix = self._compute_confusion_matrix(y_true, y_labels, n_classes)

        self.tp = np.diag(self.confusion_matrix).astype(float)
        self.fp = (np.sum(self.confusion_matrix, axis=0) - self.tp).astype(float)
        self.fn = (np.sum(self.confusion_matrix, axis=1) - self.tp).astype(float)
        self.tn = (np.sum(self.confusion_matrix) - (self.tp + self.fp + self.fn)).astype(float)

        total = float(np.sum(self.confusion_matrix))
        self.accuracy = float(np.sum(self.tp) / total) if total > 0 else 0.0

        precision_per_class = self._safe_divide(self.tp, self.tp + self.fp)
        recall_per_class = self._safe_divide(self.tp, self.tp + self.fn)
        specificity_per_class = self._safe_divide(self.tn, self.tn + self.fp)
        f1_per_class = self._safe_divide(2 * precision_per_class * recall_per_class, precision_per_class + recall_per_class)

        self.precision = float(np.mean(precision_per_class))
        self.recall = float(np.mean(recall_per_class))
        self.sensitivity = self.recall
        self.specificity = float(np.mean(specificity_per_class))
        self.f1_score = float(np.mean(f1_per_class))

    def _compute_confusion_matrix(self, y: np.ndarray, y_pred: np.ndarray, classes: int) -> np.ndarray:
        if np.any((y < 0) | (y >= classes)):
            raise ValueError(f"true label out of range for n_classes={classes}")
        if np.any((y_pred < 0) | (y_pred >= classes)):
            raise ValueError(f"predicted label out of range for n_classes={classes}")

        matrix = np.zeros((classes, classes), dtype=int)
        for true, pred in zip(y, y_pred):
            matrix[int(true), int(pred)] += 1

        return matrix

    def _regression_repr(self) -> str:
        return (
            f"\n{ansi.BOLD}{ansi.BLUE}-> regression metrics{ansi.RESET}\n"
            f"   |> MAE:  {np.array2string(np.asarray(self.mae), precision=4, floatmode='fixed')}\n"
            f"   |> MSE:  {np.array2string(np.asarray(self.mse), precision=4, floatmode='fixed')}\n"
            f"   |> RMSE: {np.array2string(np.asarray(self.rmse), precision=4, floatmode='fixed')}\n"
            f"   |> R2:   {np.array2string(np.asarray(self.r2), precision=4, floatmode='fixed')}"
        )

    def _classification_repr(self) -> str:
        return (
            f"\n{ansi.BOLD}{ansi.BLUE}-> classification metrics{ansi.RESET}\n"
            f"   |> accuracy:    {self.accuracy:.4f}\n"
            f"   |> precision:   {self.precision:.4f}\n"
            f"   |> recall:      {self.recall:.4f}\n"
            f"   |> sensitivity: {self.sensitivity:.4f}\n"
            f"   |> specificity: {self.specificity:.4f}\n"
            f"   |> f1 score:    {self.f1_score:.4f}"
        )

    def __repr__(self) -> str:
        if self.task == "regression":
            return self._regression_repr()
        if self.task in ("classification", "signal_classification"):
            return self._classification_repr()
        raise ValueError("Invalid task type")
