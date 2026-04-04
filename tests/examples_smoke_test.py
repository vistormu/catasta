import os
import subprocess
import sys
import unittest
from pathlib import Path


class ExampleSmokeTests(unittest.TestCase):
    def test_representative_examples(self):
        root = Path(__file__).resolve().parents[1]
        examples_dir = root / "examples"

        examples = [
            "03-regression_with_fnn.py",
            "11-image_classification_with_fnn.py",
            "19-signal_classification_with_fnn.py",
            "25-signal_classification_with_gp_binary_classifier.py",
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        env["CATASTA_SMOKE"] = "1"

        for example_name in examples:
            with self.subTest(example=example_name):
                run = subprocess.run(
                    [sys.executable, example_name],
                    cwd=examples_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if run.returncode != 0:
                    self.fail(
                        f"{example_name} failed with exit code {run.returncode}\n"
                        f"stdout:\n{run.stdout}\n"
                        f"stderr:\n{run.stderr}\n"
                    )


if __name__ == "__main__":
    unittest.main()
