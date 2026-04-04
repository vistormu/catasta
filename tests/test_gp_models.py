import unittest

import torch
from gpytorch.kernels import RBFKernel
from gpytorch.means import ZeroMean
from gpytorch.likelihoods import BernoulliLikelihood, GaussianLikelihood

from catasta.models import GPRegressor, GPHeadRegressor, GPBinaryClassifier, PGLikelihood


class GPModelConfigTests(unittest.TestCase):
    def test_gp_regressor_accepts_string_specs(self):
        model = GPRegressor(
            n_inducing_points=4,
            n_inputs=3,
            n_outputs=1,
            kernel="rbf",
            mean="zero",
            likelihood="gaussian",
        )
        self.assertIsNotNone(model.covar_module)
        self.assertIsNotNone(model.mean_module)
        self.assertIsNotNone(model.likelihood)

    def test_gp_regressor_accepts_class_specs(self):
        model = GPRegressor(
            n_inducing_points=4,
            n_inputs=3,
            n_outputs=1,
            kernel=RBFKernel,
            mean=ZeroMean,
            likelihood=GaussianLikelihood,
        )
        self.assertIsNotNone(model.covar_module)
        self.assertIsNotNone(model.mean_module)
        self.assertIsInstance(model.likelihood, GaussianLikelihood)

    def test_gp_head_regressor_accepts_class_specs(self):
        pre_model = torch.nn.Linear(3, 3)
        model = GPHeadRegressor(
            pre_model=pre_model,
            pre_model_output_dim=3,
            n_inputs=3,
            n_outputs=1,
            n_inducing_points=4,
            kernel=RBFKernel,
            mean=ZeroMean,
            likelihood=GaussianLikelihood,
        )
        self.assertIsNotNone(model.covar_module)
        self.assertIsNotNone(model.mean_module)
        self.assertIsInstance(model.likelihood, GaussianLikelihood)

    def test_gp_head_regressor_allows_multi_output(self):
        pre_model = torch.nn.Linear(3, 3)
        model = GPHeadRegressor(
            pre_model=pre_model,
            pre_model_output_dim=3,
            n_inputs=3,
            n_outputs=2,
            n_inducing_points=4,
            kernel="rbf",
            mean="zero",
            likelihood="gaussian",
        )
        self.assertIsNotNone(model)

    def test_gp_regressor_rejects_invalid_class(self):
        with self.assertRaisesRegex(ValueError, "kernel class must inherit"):
            GPRegressor(
                n_inducing_points=4,
                n_inputs=3,
                n_outputs=1,
                kernel=torch.nn.Linear,  # type: ignore[arg-type]
                mean="zero",
                likelihood="gaussian",
            )

    def test_gp_binary_classifier_defaults(self):
        model = GPBinaryClassifier(
            n_inducing_points=4,
            n_inputs=3,
        )
        self.assertIsNotNone(model.mean_module)
        self.assertIsNotNone(model.covar_module)
        self.assertIsInstance(model.likelihood, PGLikelihood)

    def test_gp_binary_classifier_accepts_class_specs(self):
        model = GPBinaryClassifier(
            n_inducing_points=4,
            n_inputs=3,
            kernel=RBFKernel,
            mean=ZeroMean,
            likelihood=BernoulliLikelihood,
        )
        self.assertIsNotNone(model.covar_module)
        self.assertIsNotNone(model.mean_module)
        self.assertIsInstance(model.likelihood, BernoulliLikelihood)


if __name__ == "__main__":
    unittest.main()
