from typing import NamedTuple
from datetime import datetime

import pandas as pd

from .trial_info import TrialInfo
from ..log import ansi


class OptimizationInfo(NamedTuple):
    """Dataclass containing the optimization information.

    Attributes
    ----------
    best_hyperparameters : dict
        Best hyperparameters found.
    best_metric : float
        Best metric found.
    best_trial_number : int
        Trial number of the best trial.
    duration : float
        Time elapsed during the optimization.
    history : None
        History of the trials.

    Methods
    -------
    to_pandas()
        Convert the OptimizationInfo to a pandas DataFrame.
    """
    best_hyperparameters: dict
    best_metric: float
    best_trial_number: int
    datetime_start: datetime
    datetime_end: datetime
    duration: float
    history: list[TrialInfo]

    def __repr__(self) -> str:
        msg = f"\n{ansi.BOLD}{ansi.BLUE}-> optimization results{ansi.RESET}\n"
        msg += f"   |> trial: {self.best_trial_number}\n"
        msg += f"   |> metric: {self.best_metric:.4f}\n"
        msg += f"   |> hyperparameters:\n"
        for key, value in self.best_hyperparameters.items():
            msg += f"      * {key}: {value}\n"

        return msg.rstrip()

    def to_pandas(self) -> pd.DataFrame:
        """Convert the OptimizationInfo to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the optimization information.
        """
        data = {
            "trial": [trial.number for trial in self.history],
            "metric": [trial.metric for trial in self.history],
            "datetime_start": [trial.datetime_start for trial in self.history],
            "datetime_end": [trial.datetime_end for trial in self.history],
            "duration": [trial.duration for trial in self.history],
        }

        for key in self.history[0].hyperparameters.keys():
            data[key] = [trial.hyperparameters[key] for trial in self.history]

        return pd.DataFrame(data)
