import unittest
import importlib

import numpy as np

from catasta.transformations import Identity, Transformation, Normalization, Slicing


class TransformationTests(unittest.TestCase):
    def test_identity_is_a_transformation(self):
        transform = Identity()
        self.assertIsInstance(transform, Transformation)

        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(transform(x), x)

    def test_normalization_handles_zero_denominator_edges(self):
        x = np.zeros((4,), dtype=float)
        methods = ["mad", "maxmin", "minmax", "std", "var", "zscore", "quantile", "robust"]

        for method in methods:
            y = Normalization(method)(x)
            self.assertTrue(np.all(np.isfinite(y)), msg=f"non-finite values for {method}")

    def test_slicing_amount_zero_and_negative_validation(self):
        x = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(Slicing(amount=0, end="left")(x), x)
        np.testing.assert_array_equal(Slicing(amount=2, end="left")(x), np.array([3, 4]))
        np.testing.assert_array_equal(Slicing(amount=2, end="right")(x), np.array([1, 2]))

        with self.assertRaisesRegex(ValueError, "non-negative"):
            Slicing(amount=-1, end="left")

    def test_removed_transformations_are_not_importable(self):
        module = importlib.import_module("catasta.transformations")
        self.assertFalse(hasattr(module, "KalmanFilter"))
        self.assertFalse(hasattr(module, "FIRFiltering"))

        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("catasta.transformations.kalman_filter")
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("catasta.transformations.fir_filtering")


if __name__ == "__main__":
    unittest.main()
