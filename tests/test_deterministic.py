import importlib
import unittest

import torch

deterministic_module = importlib.import_module("catasta.utils.set_deterministic")


class DeterministicTests(unittest.TestCase):
    def test_import_has_no_cudnn_side_effects(self):
        original_deterministic = torch.backends.cudnn.deterministic
        original_benchmark = torch.backends.cudnn.benchmark

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        importlib.reload(deterministic_module)

        self.assertFalse(torch.backends.cudnn.deterministic)
        self.assertTrue(torch.backends.cudnn.benchmark)

        torch.backends.cudnn.deterministic = original_deterministic
        torch.backends.cudnn.benchmark = original_benchmark

    def test_set_deterministic_sets_expected_flags(self):
        try:
            deterministic_module.set_deterministic()
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
        finally:
            torch.use_deterministic_algorithms(False)


if __name__ == "__main__":
    unittest.main()
