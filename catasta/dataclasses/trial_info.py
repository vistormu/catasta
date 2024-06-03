from typing import NamedTuple

from datetime import datetime


class TrialInfo(NamedTuple):
    """Dataclass containing the trial information.

    Attributes
    ----------
    trial_number : int
        Number of the trial.
    hyperparameters : dict
        Hyperparameters of the trial.
    metric : float
        Metric value of the trial.
    datetime_start : datetime
        Start datetime of the trial.
    datetime_end : datetime
        End datetime of the trial.
    duration : float
        Duration of the trial.
    """
    number: int
    hyperparameters: dict
    metric: float
    datetime_start: datetime
    datetime_end: datetime
    duration: float
