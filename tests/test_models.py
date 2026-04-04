import unittest

import torch

from catasta.models import (
    CNNRegressor,
    LSTMRegressor,
    TCNRegressor,
    ResNet1DRegressor,
    FeedforwardRegressor,
    MultiHeadedFeedforwardRegressor,
    MultiHeadedTransformerRegressor,
    CNNImageClassifier,
    TransformerRegressor,
    TransformerFFTRegressor,
    MambaRegressor,
    MambaFFTRegressor,
)


class ModelContractTests(unittest.TestCase):
    def test_feedforward_regressor_no_hidden_layers_respects_output_dims(self):
        x = torch.randn(5, 4)

        single = FeedforwardRegressor(n_inputs=4, n_outputs=1, dropout=0.0, hidden_dims=[])
        multi = FeedforwardRegressor(n_inputs=4, n_outputs=3, dropout=0.0, hidden_dims=[])

        self.assertEqual(single(x).shape, (5,))
        self.assertEqual(multi(x).shape, (5, 3))

    def test_cnn_image_classifier_supports_empty_feedforward_dims(self):
        model = CNNImageClassifier(
            input_shape=(8, 8, 3),
            n_classes=4,
            conv_out_channels=[4],
            feedforward_dims=[],
        )
        x = torch.randn(2, 8, 8, 3)
        y = model(x)
        self.assertEqual(y.shape, (2, 4))

    def test_patch_regressor_guards(self):
        regressor_builders = [
            lambda: TransformerRegressor(
                n_inputs=8,
                n_outputs=1,
                n_patches=0,
                d_model=4,
                n_layers=1,
                n_heads=1,
                feedforward_dim=8,
                head_dim=4,
            ),
            lambda: TransformerFFTRegressor(
                n_inputs=8,
                n_outputs=1,
                n_patches=9,
                d_model=4,
                n_layers=1,
                n_heads=1,
                feedforward_dim=8,
                head_dim=4,
            ),
            lambda: MambaRegressor(
                n_inputs=8,
                n_outputs=1,
                n_patches=3,
                d_model=4,
                d_state=2,
                d_conv=2,
                expand=1,
                n_layers=1,
            ),
            lambda: MambaFFTRegressor(
                n_inputs=8,
                n_outputs=1,
                n_patches=3,
                d_model=4,
                d_state=2,
                d_conv=2,
                expand=1,
                n_layers=1,
            ),
        ]

        for builder in regressor_builders:
            with self.assertRaises(ValueError):
                builder()

    def test_regressor_squeeze_behavior_for_multi_output(self):
        x = torch.randn(3, 8)

        models = [
            TransformerRegressor(
                n_inputs=8,
                n_outputs=2,
                n_patches=2,
                d_model=4,
                n_layers=1,
                n_heads=1,
                feedforward_dim=8,
                head_dim=4,
            ),
            TransformerFFTRegressor(
                n_inputs=8,
                n_outputs=2,
                n_patches=2,
                d_model=4,
                n_layers=1,
                n_heads=1,
                feedforward_dim=8,
                head_dim=4,
            ),
            MambaRegressor(
                n_inputs=8,
                n_outputs=2,
                n_patches=2,
                d_model=4,
                d_state=2,
                d_conv=2,
                expand=1,
                n_layers=1,
            ),
            MambaFFTRegressor(
                n_inputs=8,
                n_outputs=2,
                n_patches=2,
                d_model=4,
                d_state=2,
                d_conv=2,
                expand=1,
                n_layers=1,
            ),
        ]

        for model in models:
            self.assertEqual(model(x).shape, (3, 2))

    def test_multiheaded_feedforward_regressor_shape(self):
        x = torch.randn(4, 8)
        model = MultiHeadedFeedforwardRegressor(
            n_inputs=8,
            n_outputs=2,
            hidden_dims=[16],
            dropout=0.0,
        )
        y = model(x)
        self.assertEqual(y.shape, (4, 2))

    def test_multiheaded_transformer_regressor_shape(self):
        x = torch.randn(4, 8)
        model = MultiHeadedTransformerRegressor(
            n_inputs=8,
            n_outputs=2,
            n_patches=2,
            d_model=4,
            n_layers=1,
            n_heads=1,
            feedforward_dim=8,
            head_dim=4,
        )
        y = model(x)
        self.assertEqual(y.shape, (4, 2))

    def test_cnn_regressor_shape(self):
        x = torch.randn(5, 32)
        single = CNNRegressor(
            n_inputs=32,
            n_outputs=1,
            conv_out_channels=[8, 16],
            feedforward_dims=[16],
            dropout=0.0,
        )
        multi = CNNRegressor(
            n_inputs=32,
            n_outputs=3,
            conv_out_channels=[8, 16],
            feedforward_dims=[],
            dropout=0.0,
        )
        self.assertEqual(single(x).shape, (5,))
        self.assertEqual(multi(x).shape, (5, 3))

    def test_lstm_regressor_shape(self):
        x = torch.randn(6, 24)
        single = LSTMRegressor(
            n_inputs=24,
            n_outputs=1,
            hidden_dim=8,
            n_layers=1,
            dropout=0.0,
        )
        multi = LSTMRegressor(
            n_inputs=24,
            n_outputs=2,
            hidden_dim=8,
            n_layers=2,
            bidirectional=True,
            feedforward_dims=[16],
            dropout=0.0,
        )
        self.assertEqual(single(x).shape, (6,))
        self.assertEqual(multi(x).shape, (6, 2))

    def test_tcn_regressor_shape(self):
        x = torch.randn(4, 32)
        single = TCNRegressor(
            n_inputs=32,
            n_outputs=1,
            channel_sizes=[8, 16],
            feedforward_dims=[],
            dropout=0.0,
        )
        multi = TCNRegressor(
            n_inputs=32,
            n_outputs=3,
            channel_sizes=[8, 16, 16],
            feedforward_dims=[12],
            dropout=0.0,
        )
        self.assertEqual(single(x).shape, (4,))
        self.assertEqual(multi(x).shape, (4, 3))

    def test_resnet1d_regressor_shape(self):
        x = torch.randn(4, 64)
        single = ResNet1DRegressor(
            n_inputs=64,
            n_outputs=1,
            channels=[8, 16],
            blocks_per_stage=1,
            feedforward_dims=[],
            dropout=0.0,
        )
        multi = ResNet1DRegressor(
            n_inputs=64,
            n_outputs=2,
            channels=[8, 16, 32],
            blocks_per_stage=1,
            feedforward_dims=[16],
            dropout=0.0,
        )
        self.assertEqual(single(x).shape, (4,))
        self.assertEqual(multi(x).shape, (4, 2))

    def test_new_regressors_are_scriptable(self):
        x_short = torch.randn(2, 24)
        x_long = torch.randn(2, 64)

        models = [
            CNNRegressor(
                n_inputs=24,
                n_outputs=2,
                conv_out_channels=[4, 8],
                feedforward_dims=[8],
                dropout=0.0,
            ),
            LSTMRegressor(
                n_inputs=24,
                n_outputs=2,
                hidden_dim=8,
                n_layers=1,
                dropout=0.0,
            ),
            TCNRegressor(
                n_inputs=24,
                n_outputs=2,
                channel_sizes=[4, 8],
                feedforward_dims=[],
                dropout=0.0,
            ),
            ResNet1DRegressor(
                n_inputs=64,
                n_outputs=2,
                channels=[8, 16],
                blocks_per_stage=1,
                feedforward_dims=[],
                dropout=0.0,
            ),
        ]

        for model in models:
            scripted = torch.jit.script(model)
            sample = x_long if isinstance(model, ResNet1DRegressor) else x_short
            y = scripted(sample)
            self.assertEqual(y.shape[0], sample.shape[0])


if __name__ == "__main__":
    unittest.main()
