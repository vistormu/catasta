import unittest

import numpy as np

from catasta.dataclasses.eval_info import EvalInfo


class EvalInfoTests(unittest.TestCase):
    def test_classification_accepts_logits_and_label_arrays(self):
        y_true = np.array([0, 1, 1, 2])

        logits = np.array(
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        labels = np.array([0, 1, 1, 2])

        logits_eval = EvalInfo(task="classification", true_output=y_true, predicted_output=logits)
        labels_eval = EvalInfo(task="classification", true_output=y_true, predicted_output=labels)

        self.assertAlmostEqual(logits_eval.accuracy, 1.0)
        self.assertAlmostEqual(labels_eval.accuracy, 1.0)
        self.assertEqual(logits_eval.confusion_matrix.shape, (3, 3))

    def test_classification_fails_fast_for_invalid_labels(self):
        y_true = np.array([0, 1, 5])
        y_pred = np.array([0, 1, 1])

        with self.assertRaisesRegex(ValueError, "out of range"):
            EvalInfo(task="classification", true_output=y_true, predicted_output=y_pred, n_classes=3)

    def test_regression_r2_safe_when_targets_constant(self):
        y_true = np.array([2.0, 2.0, 2.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        info = EvalInfo(task="regression", true_output=y_true, predicted_output=y_pred)

        self.assertEqual(info.r2, 0.0)

    def test_signal_classification_task_is_supported(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])

        info = EvalInfo(task="signal_classification", true_output=y_true, predicted_output=y_pred)
        self.assertGreaterEqual(info.accuracy, 0.0)

    def test_multi_output_regression_metrics_are_per_channel(self):
        y_true = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        y_pred = np.array([[1.0, 1.0], [2.0, 5.0], [3.0, 7.0]])
        true_input = np.array([[0.1], [0.2], [0.3]])

        info = EvalInfo(
            task="regression",
            true_input=true_input,
            true_output=y_true,
            predicted_output=y_pred,
        )
        self.assertEqual(info.mae.shape, (2,))
        self.assertEqual(info.mse.shape, (2,))
        self.assertEqual(info.rmse.shape, (2,))
        self.assertEqual(info.r2.shape, (2,))
        self.assertEqual(info.true_input.shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
