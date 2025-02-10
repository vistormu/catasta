import numpy as np

from ..log import ansi


class EvalInfo:
    """A class that calculates and stores evaluation metrics for regression and classification tasks.

    Attributes
    ----------
    self.task : str
        The task type, either 'regression' or 'classification'.
    self.true_output : np.ndarray
        The true output values.
    self.predicted_output : np.ndarray
        The predicted output values.
    self.predicted_std : np.ndarray
        The standard deviation of the predicted output values.
    self.mae : float
        The mean absolute error. Only available for regression tasks.
    self.mse : float
        The mean squared error. Only available for regression tasks.
    self.rmse : float
        The root mean squared error. Only available for regression tasks.
    self.r2 : float
        The R2 score. Only available for regression tasks.
    self.confusion_matrix : np.ndarray
        The confusion matrix. Only available for classification tasks.
    self.tp : np.ndarray
        The true positive values. Only available for classification tasks.
    self.fp : np.ndarray
        The false positive values. Only available for classification tasks.
    self.fn : np.ndarray
        The false negative values. Only available for classification tasks.
    self.tn : np.ndarray
        The true negative values. Only available for classification tasks.
    self.accuracy : float
        The accuracy score. Only available for classification tasks.
    self.precision : float
        The precision score. Only available for classification tasks.
    self.recall : float
        The recall score. Only available for classification tasks.
    self.sensitivity : float
        The sensitivity score. Only available for classification tasks.
    self.specificity : float
        The specificity score. Only available for classification tasks.
    self.f1_score : float
        The F1 score. Only available for classification tasks.
    """

    def __init__(self,
                 task: str,
                 true_output: np.ndarray,
                 predicted_output: np.ndarray,
                 predicted_std: np.ndarray | None = None,
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
        predicted_std : np.ndarray, optional
            The standard deviation of the predicted output values. If not provided, it is set to an array of zeros.
        n_classes : int, optional
            The number of classes in the classification task. If not provided, it is set to the number of unique classes.
        """

        self.task = task
        self.true_output = true_output
        self.predicted_output = predicted_output
        self.predicted_std = predicted_std if predicted_std is not None else np.zeros_like(predicted_output)

        if self.task == 'regression':
            self._calculate_regression_metrics(self.true_output, self.predicted_output)
        elif self.task == 'classification':
            self._calculate_classification_metrics(self.true_output, self.predicted_output, n_classes)
        else:
            raise ValueError('Invalid task type')

    def _calculate_regression_metrics(self, y: np.ndarray, y_pred: np.ndarray) -> None:
        self.mae: float = np.mean(np.abs(y - y_pred)).astype(float)
        self.mse: float = np.mean((y - y_pred)**2).astype(float)
        self.rmse: float = np.sqrt(np.mean((y - y_pred)**2)).astype(float)
        self.r2: float = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2).astype(float)

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
        return f"\n{ansi.BOLD}{ansi.BLUE}-> regression metrics{ansi.RESET}\n" + \
            f"   |> MAE:  {self.mae:.4f}\n" + \
            f"   |> MSE:  {self.mse:.4f}\n" + \
            f"   |> RMSE: {self.rmse:.4f}\n" + \
            f"   |> R2:   {self.r2:.4f}"

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
