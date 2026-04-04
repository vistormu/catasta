import unittest

import catasta
from catasta import models


class PublicAPIContractTests(unittest.TestCase):
    def test_root_exports(self):
        expected_exports = {
            "Scaffold",
            "Dataset",
            "Archway",
            "Foundation",
        }
        for export_name in expected_exports:
            self.assertTrue(
                hasattr(catasta, export_name),
                msg=f"missing catasta export: {export_name}",
            )

    def test_models_exports(self):
        expected_model_exports = {
            "GPRegressor",
            "FeedforwardRegressor",
            "CNNRegressor",
            "LSTMRegressor",
            "TCNRegressor",
            "ResNet1DRegressor",
            "MultiHeadedFeedforwardRegressor",
            "TransformerRegressor",
            "TransformerFFTRegressor",
            "MultiHeadedTransformerRegressor",
            "MambaRegressor",
            "MambaFFTRegressor",
            "GPHeadRegressor",
            "GPBinaryClassifier",
            "PGLikelihood",
            "FeedforwardImageClassifier",
            "TransformerImageClassifier",
            "TransformerFFTImageClassifier",
            "MambaImageClassifier",
            "MambaFFTImageClassifier",
            "CNNImageClassifier",
        }
        for export_name in expected_model_exports:
            self.assertTrue(
                hasattr(models, export_name),
                msg=f"missing catasta.models export: {export_name}",
            )


if __name__ == "__main__":
    unittest.main()
