from typing import Callable
from datetime import datetime

import numpy as np

import optuna
from optuna import Trial, Study
from optuna.trial import FrozenTrial

from ..dataclasses import OptimizationInfo, TrialInfo
from .utils import OptimizationLogger, get_sampler
from ..log import ansi


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

        if not self.verbose:
            return

        # log optimization info
        n_hyperparameters = len(self.hyperparameter_space)
        n_combinations = np.prod([len(values) for values in self.hyperparameter_space.values()])
        print(
            f"{ansi.BOLD}{ansi.GREEN}-> optimization info{ansi.RESET}\n"
            f"   |> hyperparameters: {n_hyperparameters} ({n_combinations} combinations)\n"
            f"   |> sampler:         {self.sampler.__class__.__name__}\n"
            f"   |> direction:       {self.direction}"
        )

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
                if isinstance(e, KeyboardInterrupt):
                    print(
                        f"{ansi.BOLD}{ansi.YELLOW}-> optimization interrupted\n"
                        f"   |> trial: {trial.number+1}{ansi.RESET}"
                    )
                    raise e

                if self.catch_exceptions:
                    print(
                        f"{ansi.BOLD}{ansi.RED}-> exception caught\n"
                        f"   |> trial: {trial.number+1}\n"
                        f"   |> error: {e}{ansi.RESET}"
                    )
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

            print(optimization_logger) if self.verbose else None

        study = optuna.create_study(
            sampler=self.sampler,
            direction=self.direction,
        )
        study.optimize(
            func=optimize_optuna,
            n_trials=self.n_trials,
            callbacks=[on_trial_complete],
        )

        first_datetime = datetime.now()
        for trial in study.trials:
            if trial.datetime_start is not None:
                first_datetime = trial.datetime_start
                break

        last_datetime = datetime.now()
        for trial in reversed(study.trials):
            if trial.datetime_start is not None:
                last_datetime = trial.datetime_start
                break

        history = []
        for trial in study.trials:
            metric = np.inf if self.direction == "minimize" else -np.inf
            if trial.value is not None:
                metric = trial.value

            history.append(TrialInfo(
                number=trial.number+1,
                hyperparameters=trial.params,
                metric=metric,
                datetime_start=trial.datetime_start,
                datetime_end=trial.datetime_complete,
                duration=trial.duration.total_seconds(),
            ))

        return OptimizationInfo(
            best_hyperparameters=study.best_params,
            best_metric=study.best_value,
            best_trial_number=study.best_trial.number+1,
            datetime_start=first_datetime,
            datetime_end=last_datetime,
            duration=last_datetime.timestamp() - first_datetime.timestamp(),
            history=history,
        )
