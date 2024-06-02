from typing import NamedTuple


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
    time_elapsed : float
        Time elapsed during the optimization.
    trial_history : None
        History of the trials.
    """
    best_hyperparameters: dict
    best_metric: float
    best_trial_number: int
    time_elapsed: float
    trial_history: None
