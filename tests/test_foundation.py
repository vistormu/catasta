import unittest

from catasta.foundation.foundation import Foundation


class FoundationTests(unittest.TestCase):
    def test_invalid_direction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid direction"):
            Foundation(
                hyperparameter_space={"x": (1, 2)},
                objective_function=lambda params, extra: float(params["x"]),
                sampler="random",
                n_trials=1,
                direction="sideways",
                verbose=False,
            )

    def test_optimize_runs_and_returns_history(self):
        foundation = Foundation(
            hyperparameter_space={"x": (1, 2), "y": (3, 4)},
            objective_function=lambda params, extra: float(params["x"] + params["y"]),
            sampler="random",
            n_trials=2,
            direction="minimize",
            verbose=False,
        )

        info = foundation.optimize()

        self.assertEqual(len(info.history), 2)
        self.assertIn("x", info.best_hyperparameters)
        self.assertGreaterEqual(info.duration, 0.0)

    def test_extra_dict_is_copied_per_trial_and_not_mutated(self):
        extra = {"fixed": 10}

        def objective(params: dict, objective_extra_parameters: dict) -> float:
            objective_extra_parameters["fixed"] = -1
            return float(params["x"])

        foundation = Foundation(
            hyperparameter_space={"x": (1, 2)},
            objective_function=objective,
            objective_extra_parameters=extra,
            sampler="random",
            n_trials=2,
            direction="minimize",
            verbose=False,
        )
        foundation.optimize()

        self.assertEqual(extra["fixed"], 10)


if __name__ == "__main__":
    unittest.main()
