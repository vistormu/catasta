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
        """Initialize the Foundation class.

        Arguments
        ---------
        hyperparameter_space : dict[str, tuple]
            Dictionary containing the hyperparameters to optimize. The keys are the hyperparameter names and the values are tuples containing the hyperparameters search space.
        objective_function : Callable[[dict], float]
            Objective function to optimize. It receives a dictionary containing the hyperparameters as input and returns a float value.
        sampler : str
            Name of the sampler to use. Valid values are "random", "rs", "tpe", "gp", and "bogp".
        n_trials : int
            Number of trials to run.
        direction : str
            Direction of the optimization. Valid values are "minimize" and "maximize".
        use_secretary : bool, optional
            Whether to use the secretary algorithm. The secretary algorithm will reject the first n_trials/e trials and select the best trial after that. Default is False.
        catch_exceptions : bool, optional
            Whether to catch exceptions raised by the objective function. Default is False.
        verbose : bool, optional
            Whether to print optimization information. Default is True.

        Raises
        ------
        ValueError
            If the direction is invalid.
            If the sampler is invalid.
        """
        self.hyperparameter_space = hyperparameter_space
        self.objective_function = objective_function
        self.sampler = get_sampler(sampler)
        self.n_trials = n_trials
        self.direction = direction
        self.use_secretary = use_secretary
        self.catch_exceptions = catch_exceptions

        self.verbose = verbose

        if direction not in ["minimize", "maximize"]:
            raise ValueError("Invalid direction")

        self.logger = Logger("catasta", disable=not verbose)
        self._log_optimization_info()

    def _log_optimization_info(self) -> None:
        n_hyperparameters = len(self.hyperparameter_space)
        n_combinations = np.prod([len(values) for values in self.hyperparameter_space.values()])
        self.logger.info(f"""    OPTIMIZATION INFO
    -> hyperparameters: {n_hyperparameters} ({n_combinations} combinations)
    -> sampler:         {self.sampler.__class__.__name__}
    -> direction:       {self.direction}""")

    def optimize(self) -> OptimizationInfo:
        """Optimize the hyperparameters.

        Returns
        -------
        OptimizationInfo
            Object containing the information about the optimization.

        Raises
        ------
        Exception
            If the objective function raises an exception and catch_exceptions is False.
        """
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
            if self.use_secretary and trial.number > self.n_trials/np.e and study.best_trial.number == trial.number:
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
