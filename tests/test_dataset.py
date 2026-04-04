import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from catasta.dataset.catasta_dataset import CatastaDataset, scan_splits


class DatasetTests(unittest.TestCase):
    def test_scan_splits_requires_explicit_alias(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "train"))
            os.makedirs(os.path.join(tmpdir, "val"))

            with self.assertRaisesRegex(ValueError, "allow_split_alias=True"):
                scan_splits(tmpdir, allow_alias=False)

            train, val, test = scan_splits(tmpdir, allow_alias=True)
            self.assertEqual((train, val, test), ("train", "val", "val"))

    def test_regression_subset_uses_only_real_csv_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for split in ("train", "val", "test"):
                split_dir = os.path.join(tmpdir, split)
                os.makedirs(split_dir)

                pd.DataFrame({"input": [1.0, 2.0], "output": [3.0, 4.0]}).to_csv(
                    os.path.join(split_dir, "a.csv"), index=False
                )
                with open(os.path.join(split_dir, "ignored.csv.bak"), "w", encoding="utf-8") as f:
                    f.write("input,output\n1,2\n")

            dataset = CatastaDataset(
                root=tmpdir,
                task="regression",
                input_name="input",
                output_name="output",
            )
            self.assertEqual(len(dataset.train), 2)
            self.assertEqual(len(dataset.validation), 2)
            self.assertEqual(len(dataset.test), 2)

    def test_classification_split_checks_class_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for split in ("train", "val", "test"):
                os.makedirs(os.path.join(tmpdir, split), exist_ok=True)

            for class_name in ("0", "1"):
                os.makedirs(os.path.join(tmpdir, "train", class_name), exist_ok=True)
                os.makedirs(os.path.join(tmpdir, "val", class_name), exist_ok=True)

            os.makedirs(os.path.join(tmpdir, "test", "0"), exist_ok=True)

            image = np.zeros((2, 2, 3), dtype=np.uint8)
            Image.fromarray(image).save(os.path.join(tmpdir, "train", "0", "a.png"))
            Image.fromarray(image).save(os.path.join(tmpdir, "train", "1", "b.png"))
            Image.fromarray(image).save(os.path.join(tmpdir, "val", "0", "c.png"))
            Image.fromarray(image).save(os.path.join(tmpdir, "val", "1", "d.png"))
            Image.fromarray(image).save(os.path.join(tmpdir, "test", "0", "e.png"))

            with self.assertRaisesRegex(ValueError, "number of classes in train and test"):
                CatastaDataset(root=tmpdir, task="classification")

    def test_signal_classification_tabular_dataset_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for split in ("train", "val", "test"):
                split_dir = os.path.join(tmpdir, split)
                os.makedirs(split_dir)
                pd.DataFrame(
                    {
                        "input": [0.0, 1.0, 2.0, 3.0],
                        "label": [0, 1, 0, 1],
                    }
                ).to_csv(os.path.join(split_dir, "data.csv"), index=False)

            dataset = CatastaDataset(
                root=tmpdir,
                task="signal_classification",
                input_name="input",
                output_name="label",
            )

            x, y = dataset.train[0]
            self.assertEqual(x.shape, (1,))
            self.assertIsInstance(int(y.item()), int)

    def test_signal_classification_rejects_unknown_split_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "train"))
            os.makedirs(os.path.join(tmpdir, "val"))
            os.makedirs(os.path.join(tmpdir, "test"))

            pd.DataFrame({"input": [0.0, 1.0], "label": [0, 1]}).to_csv(
                os.path.join(tmpdir, "train", "data.csv"), index=False
            )
            pd.DataFrame({"input": [2.0], "label": [2]}).to_csv(
                os.path.join(tmpdir, "val", "data.csv"), index=False
            )
            pd.DataFrame({"input": [3.0], "label": [0]}).to_csv(
                os.path.join(tmpdir, "test", "data.csv"), index=False
            )

            with self.assertRaisesRegex(ValueError, "unknown label"):
                CatastaDataset(
                    root=tmpdir,
                    task="signal_classification",
                    input_name="input",
                    output_name="label",
                )


if __name__ == "__main__":
    unittest.main()
