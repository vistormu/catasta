import os
import tempfile
import unittest

import numpy as np
import torch

from catasta.archway.archway import Archway, _compile_model, _get_dtype


class ArchwayTests(unittest.TestCase):
    def test_get_dtype_parses_supported_values(self):
        self.assertEqual(_get_dtype("float32"), torch.float32)
        self.assertEqual(_get_dtype("qint8"), torch.qint8)

        with self.assertRaises(ValueError):
            _get_dtype("bad")

    def test_quantized_compile_requires_layers_and_dtype(self):
        model = torch.nn.Linear(4, 2)

        with self.assertRaisesRegex(ValueError, "layers"):
            _compile_model(model, "torchscript_quantized")

        with self.assertRaisesRegex(ValueError, "layers"):
            _compile_model(model, "torchscript_quantized_optimized", layers={torch.nn.Linear})

    def test_predict_torch_handles_numpy_and_tensor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            model = torch.nn.Sequential(torch.nn.Linear(3, 2))
            torch.save(model, path)

            archway = Archway(path=path, compile_method="none", device="cpu", dtype="float32")

            x_np = np.ones((2, 3), dtype=np.float32)
            y_np = archway.predict_torch(x_np)
            self.assertTrue(isinstance(y_np, torch.Tensor))
            self.assertEqual(tuple(y_np.shape), (2, 2))

            x_t = torch.ones(2, 3)
            y_t = archway.predict_torch(x_t)
            self.assertTrue(isinstance(y_t, torch.Tensor))
            self.assertEqual(tuple(y_t.shape), (2, 2))

            with self.assertRaises(TypeError):
                archway.predict_torch([1, 2, 3])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
