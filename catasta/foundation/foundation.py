from typing import Callable

import numpy as np

import optuna
from optuna import Trial, Study
from optuna.trial import FrozenTrial

from vclog import Logger

from ..dataclasses import OptimizationInfo
from .utils import OptimizationLogger, get_sampler


class Foundation:
    def __init__(self, *,
                 hyperparameter_space: dict[str, tuple],
                 objective_function: Callable[[dict], float],
                 sampler: str,
                 n_trials: int,
                 direction: str,
                 use_secretary: bool = False,
                 catch_exceptions: bool = False,
                 verbose: bool = True,
                 ) -> None:
        self.hyperparameter_space = hyperparameter_space
        self.objective_function = objective_function
        self.sampler = get_sampler(sampler)
        self.n_trials = n_trials
        self.direction = direction
        self.use_secretary = use_secretary
        self.catch_exceptions = catch_exceptions

        self.verbose = verbose

        self.logger = Logger("catasta", disable=not verbose)
        self._log_optimization_info()

    def _log_optimization_info(self) -> None:
        n_hyperparameters = len(self.hyperparameter_space)
        n_combinations = np.prod([len(values) for values in self.hyperparameter_space.values()])
        self.logger.info(f"""    OPTIMIZATION INFO
    -> hyperparameters: {n_hyperparameters} ({n_combinations} combinations)
    -> sampler:         {self.sampler.__class__.__name__}
    -> direction:       {self.direction}""")

    def _should_stop(self, metric: float | None, best_metric: float, trial_number: int) -> bool:
        if metric is None:
            return False

        is_better = metric > best_metric if self.direction == "maximize" else metric < best_metric

        return is_better and self.use_secretary and trial_number > np.round(self.n_trials/np.e)

    def optimize(self) -> OptimizationInfo:
        def optimize_optuna(trial: Trial) -> float:
            params = {name: trial.suggest_categorical(name, values) for name, values in self.hyperparameter_space.items()}

            try:
                metric = self.objective_function(params)
            except Exception as e:
                if self.catch_exceptions:
                    self.logger.error(e)
                    metric = np.inf if self.direction == "minimize" else -np.inf
                else:
                    raise e

            return metric

        optimization_logger = OptimizationLogger(self.n_trials)

        def on_trial_complete(study: Study, trial: FrozenTrial) -> None:
            if self._should_stop(trial.value, study.best_value, trial.number+1):
                study.stop()

            optimization_logger.log(
                trial_number=trial.number+1,
                metric=trial.value if trial.value is not None else 0.0,  # TMP
                best_trial_number=study.best_trial.number+1,
                best_metric=study.best_value,
                time_elapsed=trial.duration.total_seconds() if trial.duration is not None else 0.0,  # TMP
            )

            Logger.plain(optimization_logger, color="green") if self.verbose else None

        study = optuna.create_study(
            sampler=self.sampler,
            direction=self.direction,
        )
        study.optimize(
            func=optimize_optuna,
            n_trials=self.n_trials,
            callbacks=[on_trial_complete],
        )

        return OptimizationInfo(
            best_hyperparameters=study.best_params,
        )
