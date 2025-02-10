import optuna
import time

from ...log import ansi


def format_time(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")

    return " ".join(parts)


class OptimizationLogger:
    def __init__(self, n_trials: int) -> None:
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        self.trial_number = 0
        self.best_trial_number = 0
        self.best_metric = 0.0
        self.n_trials = n_trials
        self.avg_time_per_trial = 0.0
        self.start_time = time.time()

    def log(self,
            trial_number: int,
            metric: float,
            best_trial_number: int,
            best_metric: float,
            time_elapsed: float,
            ) -> None:
        self.trial_number = trial_number
        self.metric = metric
        self.best_trial_number = best_trial_number
        self.best_metric = best_metric

        self.avg_time_per_trial = (self.avg_time_per_trial * (trial_number - 1) + time_elapsed) / trial_number

    def __repr__(self) -> str:
        # progress
        percentage = int(self.trial_number / self.n_trials * 100)

        # time
        time_remaining: int = int(self.avg_time_per_trial * (self.n_trials - self.trial_number))
        time_from_start: int = int(time.time() - self.start_time)

        # message
        clear = ansi.START + (ansi.UP + ansi.CLEAR_LINE)*4
        trial_msg = f"   |> trial:           {self.trial_number}/{self.n_trials} ({percentage}%)\n"
        last_metric = f"   |> metric:          {self.metric:.4f}\n"
        best_metric = f"   |> best metric:     {self.best_metric:.4f} (at trial {self.best_trial_number})\n"
        time_msg: str = f"   |> time:            {format_time(time_from_start)} ({format_time(time_remaining)} remaining)"

        return f"{clear if self.trial_number != 1 else ''}{trial_msg}{last_metric}{best_metric}{time_msg}"
