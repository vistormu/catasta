import numpy as np


class ClassificationEvalInfo:
    def __init__(self, *,
                 true_labels: np.ndarray,
                 predicted_labels: np.ndarray,
                 ) -> None:
        self.true_labels: np.ndarray = true_labels
        self.predicted_labels: np.ndarray = predicted_labels

        self.confusion_matrix: np.ndarray = self._compute_confusion_matrix()
        self.tp: np.ndarray = np.diag(self.confusion_matrix)
        self.fp: np.ndarray = np.sum(self.confusion_matrix, axis=0) - self.tp
        self.fn: np.ndarray = np.sum(self.confusion_matrix, axis=1) - self.tp
        self.tn: np.ndarray = np.sum(self.confusion_matrix) - (self.tp + self.fp + self.fn)

        self.accuracy: np.ndarray = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)
        self.precision: np.ndarray = self.tp / (self.tp + self.fp)
        self.sensitivity: np.ndarray = self.tp / (self.tp + self.fn)
        self.specificity: np.ndarray = self.tn / (self.tn + self.fp)
        self.f1: np.ndarray = 2 * (self.precision * self.sensitivity) / (self.precision + self.sensitivity)

    def _compute_confusion_matrix(self) -> np.ndarray:
        classes: int = len(np.unique(self.true_labels))
        matrix: np.ndarray = np.zeros((classes, classes), dtype=int)

        for true, pred in zip(self.true_labels, self.predicted_labels):
            matrix[true, pred] += 1

        return matrix

    def __repr__(self) -> str:
        accuracy: str = f"accuracy:{self.accuracy.mean(): .4f}"
        precision: str = f"precision:{self.precision.mean(): .4f}"
        sensitivity: str = f"sensitivity:{self.sensitivity.mean(): .4f}"
        specificity: str = f"specificity:{self.specificity.mean(): .4f}"
        f1: str = f"f1:{self.f1.mean(): .4f}"
        confusion_matrix: str = f"confusion_matrix:\n{self.confusion_matrix}"

        return "\n".join([accuracy, precision, sensitivity, specificity, f1, confusion_matrix])
