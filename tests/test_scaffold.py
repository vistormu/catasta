import os
import tempfile
import unittest

import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from catasta.dataset import CatastaDataset
from catasta.models import FeedforwardRegressor
from catasta.scaffold.scaffold import Scaffold
from catasta.scaffold.utils.model_state_manager import ModelStateManager


class ScaffoldAndStateTests(unittest.TestCase):
    def _make_regression_dataset_root(self, root: str) -> None:
        for split in ("train", "val", "test"):
            split_dir = os.path.join(root, split)
            os.makedirs(split_dir)
            df = pd.DataFrame({"input": [0.0, 1.0, 2.0, 3.0], "output": [0.0, 1.0, 2.0, 3.0]})
            df.to_csv(os.path.join(split_dir, "data.csv"), index=False)

    def _make_signal_classification_dataset_root(self, root: str) -> None:
        for split in ("train", "val", "test"):
            split_dir = os.path.join(root, split)
            os.makedirs(split_dir)
            df = pd.DataFrame(
                {
                    "input": [0.0, 1.0, 2.0, 3.0],
                    "label": [0, 1, 0, 1],
                }
            )
            df.to_csv(os.path.join(split_dir, "data.csv"), index=False)

    def test_save_filename_only_does_not_require_parent_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_regression_dataset_root(tmpdir)

            dataset = CatastaDataset(root=tmpdir, task="regression", input_name="input", output_name="output")
            model = FeedforwardRegressor(n_inputs=1, n_outputs=1, dropout=0.0, hidden_dims=[])
            scaffold = Scaffold(
                model=model,
                dataset=dataset,
                optimizer="sgd",
                loss_function="mse",
                device="cpu",
                verbose=False,
            )

            with tempfile.TemporaryDirectory() as save_dir:
                original_cwd = os.getcwd()
                try:
                    os.chdir(save_dir)
                    scaffold.save("model.pt")
                    self.assertTrue(os.path.exists("model.pt"))
                finally:
                    os.chdir(original_cwd)

    def test_lbfgs_training_path_uses_closure_successfully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_regression_dataset_root(tmpdir)

            dataset = CatastaDataset(root=tmpdir, task="regression", input_name="input", output_name="output")
            model = FeedforwardRegressor(n_inputs=1, n_outputs=1, dropout=0.0, hidden_dims=[])
            scaffold = Scaffold(
                model=model,
                dataset=dataset,
                optimizer="lbfgs",
                loss_function="mse",
                device="cpu",
                verbose=False,
            )

            info = scaffold.train(epochs=1, batch_size=2, lr=0.1)
            self.assertEqual(len(info.train_losses), 1)
            self.assertEqual(len(info.val_losses), 1)

    def test_model_state_manager_keeps_deep_copied_best_state(self):
        model = torch.nn.Linear(2, 1)
        manager = ModelStateManager(mode="none")

        original_weight = model.weight.detach().clone()
        manager(model, loss=0.1)

        with torch.no_grad():
            model.weight.add_(10.0)

        manager(model, loss=0.2)
        manager.load_best_model_state(model)

        self.assertTrue(torch.allclose(model.weight, original_weight))

    def test_scheduler_contract_steps_and_records_lr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_regression_dataset_root(tmpdir)

            dataset = CatastaDataset(root=tmpdir, task="regression", input_name="input", output_name="output")
            model = FeedforwardRegressor(n_inputs=1, n_outputs=1, dropout=0.0, hidden_dims=[])
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

            scaffold = Scaffold(
                model=model,
                dataset=dataset,
                optimizer=optimizer,
                loss_function="mse",
                device="cpu",
                verbose=False,
            )

            info = scaffold.train(epochs=2, batch_size=2, lr=0.1, scheduler=scheduler)
            self.assertEqual(len(info.lr_values), 2)
            self.assertLess(info.lr_values[1], info.lr_values[0])

    def test_scheduler_val_loss_contract_requires_plateau_scheduler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_regression_dataset_root(tmpdir)

            dataset = CatastaDataset(root=tmpdir, task="regression", input_name="input", output_name="output")
            model = FeedforwardRegressor(n_inputs=1, n_outputs=1, dropout=0.0, hidden_dims=[])
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

            scaffold = Scaffold(
                model=model,
                dataset=dataset,
                optimizer=optimizer,
                loss_function="mse",
                device="cpu",
                verbose=False,
            )

            info = scaffold.train(
                epochs=2,
                batch_size=2,
                lr=0.1,
                scheduler=(scheduler, "val_loss"),
            )
            self.assertEqual(len(info.train_losses), 2)

    def test_early_stopping_tuple_contract_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_regression_dataset_root(tmpdir)

            dataset = CatastaDataset(root=tmpdir, task="regression", input_name="input", output_name="output")
            model = FeedforwardRegressor(n_inputs=1, n_outputs=1, dropout=0.0, hidden_dims=[])
            scaffold = Scaffold(
                model=model,
                dataset=dataset,
                optimizer="sgd",
                loss_function="mse",
                device="cpu",
                verbose=False,
            )

            with self.assertRaisesRegex(ValueError, "trend alpha must be in \\[0, 1\\)"):
                scaffold.train(epochs=1, batch_size=2, lr=0.1, early_stopping=("trend", 1.2, 1))

            with self.assertRaisesRegex(ValueError, "trend patience must be >= 1"):
                scaffold.train(epochs=1, batch_size=2, lr=0.1, early_stopping=("trend", 0.9, 0))

            with self.assertRaisesRegex(ValueError, "plateau min_delta must be >= 0"):
                scaffold.train(epochs=1, batch_size=2, lr=0.1, early_stopping=("plateau", 2, -0.01))

            with self.assertRaisesRegex(ValueError, "early_stopping must be"):
                scaffold.train(epochs=1, batch_size=2, lr=0.1, early_stopping=(0.9, 2))

    def test_early_stopping_plateau_policy_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_regression_dataset_root(tmpdir)

            dataset = CatastaDataset(root=tmpdir, task="regression", input_name="input", output_name="output")
            model = FeedforwardRegressor(n_inputs=1, n_outputs=1, dropout=0.0, hidden_dims=[])
            scaffold = Scaffold(
                model=model,
                dataset=dataset,
                optimizer="sgd",
                loss_function="mse",
                device="cpu",
                verbose=False,
            )

            info = scaffold.train(
                epochs=3,
                batch_size=2,
                lr=0.1,
                early_stopping=("plateau", 2, 1e-4),
            )
            self.assertGreaterEqual(len(info.train_losses), 1)

    def test_signal_classification_train_and_evaluate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_signal_classification_dataset_root(tmpdir)

            dataset = CatastaDataset(
                root=tmpdir,
                task="signal_classification",
                input_name="input",
                output_name="label",
            )
            model = torch.nn.Linear(1, 2)
            scaffold = Scaffold(
                model=model,
                dataset=dataset,
                optimizer="sgd",
                loss_function="cross_entropy",
                device="cpu",
                verbose=False,
            )

            info = scaffold.train(epochs=1, batch_size=2, lr=0.1)
            self.assertEqual(len(info.train_losses), 1)

            eval_info = scaffold.evaluate(batch_size=2)
            self.assertGreaterEqual(eval_info.accuracy, 0.0)
            self.assertIsNotNone(eval_info.true_input)

    def test_evaluate_prediction_function_is_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_regression_dataset_root(tmpdir)

            dataset = CatastaDataset(root=tmpdir, task="regression", input_name="input", output_name="output")
            model = FeedforwardRegressor(n_inputs=1, n_outputs=1, dropout=0.0, hidden_dims=[])
            scaffold = Scaffold(
                model=model,
                dataset=dataset,
                optimizer="sgd",
                loss_function="mse",
                device="cpu",
                verbose=False,
            )

            scaffold.train(epochs=1, batch_size=2, lr=0.1)

            eval_info = scaffold.evaluate(
                batch_size=2,
                prediction_function=lambda model, x: model(x).detach().cpu().numpy(),
            )
            self.assertIsNotNone(eval_info.true_input)


if __name__ == "__main__":
    unittest.main()
