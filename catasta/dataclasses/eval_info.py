from typing import Any
import numpy as np

from ..log import ansi


class EvalInfo:
    """A class that calculates and stores evaluation metrics for regression and classification tasks.

    Attributes
    ----------
    task : str
        The task type, either 'regression' or 'classification'.
    true_input : np.ndarray
        The input used for the evaluation
    true_output : np.ndarray
        The true output values.
    predicted_output : Any
        The predicted output values.
    mae : np.ndarray
        The mean absolute error. Only available for regression tasks.
    mse : np.ndarray
        The mean squared error. Only available for regression tasks.
    rmse : np.ndarray
        The root mean squared error. Only available for regression tasks.
    r2 : np.ndarray
        The R2 score. Only available for regression tasks.
    confusion_matrix : np.ndarray
        The confusion matrix. Only available for classification tasks.
    tp : np.ndarray
        The true positive values. Only available for classification tasks.
    fp : np.ndarray
        The false positive values. Only available for classification tasks.
    fn : np.ndarray
        The false negative values. Only available for classification tasks.
    tn : np.ndarray
        The true negative values. Only available for classification tasks.
    accuracy : float
        The accuracy score. Only available for classification tasks.
    precision : float
        The precision score. Only available for classification tasks.
    recall : float
        The recall score. Only available for classification tasks.
    sensitivity : float
        The sensitivity score. Only available for classification tasks.
    specificity : float
        The specificity score. Only available for classification tasks.
    f1_score : float
        The F1 score. Only available for classification tasks.
    """

    def __init__(self,
                 task: str,
                 true_input: np.ndarray,
                 true_output: np.ndarray,
                 predicted_output: Any,
                 n_classes: int | None = None
                 ) -> None:
        """
        Parameters
        ----------
        task : str
            The task type, either 'regression' or 'classification'.
        true_output : np.ndarray
            The true output values.
        predicted_output : np.ndarray
            The predicted output values.
        n_classes : int, optional
            The number of classes in the classification task. If not provided, it is set to the number of unique classes.
        """

        self.task = task

        self.true_input = true_input
        self.true_output = true_output
        self.predicted_output = predicted_output

        try:
            self.predicted_output = np.asarray(self.predicted_output)
        except:
            return

        if self.task == 'regression':
            self._calculate_regression_metrics(self.true_output, self.predicted_output)
        elif self.task == 'classification':
            self._calculate_classification_metrics(self.true_output, self.predicted_output, n_classes)
        else:
            raise ValueError('Invalid task type')

    def _calculate_regression_metrics(self, y: np.ndarray, y_pred: np.ndarray) -> None:
        if y.ndim == 1:
            y = y[:, np.newaxis]
            y_pred = y_pred[:, np.newaxis]

        self.mae: np.ndarray = np.mean(np.abs(y - y_pred), axis=0)
        self.mse: np.ndarray = np.mean((y - y_pred) ** 2, axis=0)
        self.rmse: np.ndarray = np.sqrt(self.mse)
        ss_res = np.sum((y - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
        self.r2: np.ndarray = 1 - ss_res / ss_tot

    def _calculate_classification_metrics(self, y: np.ndarray, y_pred: np.ndarray, n_classes: int | None) -> None:
        y_pred = np.argmax(y_pred, axis=1)
        self.confusion_matrix: np.ndarray = self._compute_confusion_matrix(y, y_pred, n_classes)

        self.tp: np.ndarray = np.diag(self.confusion_matrix)
        self.fp: np.ndarray = np.sum(self.confusion_matrix, axis=0) - self.tp
        self.fn: np.ndarray = np.sum(self.confusion_matrix, axis=1) - self.tp
        self.tn: np.ndarray = np.sum(self.confusion_matrix) - (self.tp + self.fp + self.fn)

        tp: int = np.sum(self.tp)
        fp: int = np.sum(self.fp)
        fn: int = np.sum(self.fn)
        tn: int = np.sum(self.tn)

        self.accuracy: float = (tp + tn) / (tp + fp + fn + tn)
        self.precision: float = tp / (tp + fp)
        self.recall = self.sensitivity = tp / (tp + fn)
        self.specificity: float = tn / (tn + fp)
        self.f1_score: float = 2 * (self.precision * self.recall) / (self.precision + self.recall)

    def _compute_confusion_matrix(self, y: np.ndarray, y_pred: np.ndarray, n_classes: int | None) -> np.ndarray:
        classes: int = len(np.unique(y)) if n_classes is None else n_classes

        matrix: np.ndarray = np.zeros((classes, classes), dtype=int)
        for true, pred in zip(y, y_pred):
            matrix[true, pred] += 1

        return matrix

    def _regression_repr(self) -> str:
        return (
            f"\n{ansi.BOLD}{ansi.BLUE}-> regression metrics{ansi.RESET}\n"
            f"   |> MAE:  {np.array2string(self.mae, precision=4, floatmode='fixed')}\n"
            f"   |> MSE:  {np.array2string(self.mse, precision=4, floatmode='fixed')}\n"
            f"   |> RMSE: {np.array2string(self.rmse, precision=4, floatmode='fixed')}\n"
            f"   |> R2:   {np.array2string(self.r2, precision=4, floatmode='fixed')}"
        )

    def _classification_repr(self) -> str:
        return f"\n{ansi.BOLD}{ansi.BLUE}-> classification metrics{ansi.RESET}\n" + \
            f"   |> accuracy:    {self.accuracy:.4f}\n" + \
            f"   |> precision:   {self.precision:.4f}\n" + \
            f"   |> recall:      {self.recall:.4f}\n" + \
            f"   |> sensitivity: {self.sensitivity:.4f}\n" + \
            f"   |> specificity: {self.specificity:.4f}\n" + \
            f"   |> f1 score:    {self.f1_score:.4f}"

    def __repr__(self) -> str:
        if self.task == 'regression':
            return self._regression_repr()
        elif self.task == 'classification':
            return self._classification_repr()
        else:
            raise ValueError('Invalid task type')
