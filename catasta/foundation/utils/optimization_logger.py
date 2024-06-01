import optuna


class OptimizationLogger:
    def __init__(self, n_trials: int) -> None:
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        self.trial_number = 0
        self.best_trial_number = 0
        self.best_metric = 0.0
        self.n_trials = n_trials
        self.avg_time_per_trial = 0.0

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
        # PERCENTAGE
        percentage = self.trial_number / self.n_trials
        percentage_str = f"{int(percentage*100)}%" if percentage > 0.09 else f"0{int(percentage*100)}%"
        trial_msg = f"    -> trial:           {self.trial_number}/{self.n_trials} ({percentage_str})\n"

        # METRICS
        last_metric = f"    -> metric:          {self.metric:.4f}\n"
        best_metric = f"    -> best metric:     {self.best_metric:.4f} ({self.best_trial_number})\n"

        # TIME
        time_remaining: int = int(self.avg_time_per_trial * (self.n_trials - self.trial_number))
        time_remaining_str: str = f"{int(time_remaining/60)}m {time_remaining%60}s"
        time_msg: str = f"    -> time:            {time_remaining_str}"

        clear_line: str = "\x1b[2K" + "\r" + "\x1b[1A"  # ]]

        msg = ""
        if self.trial_number != 1:
            msg += clear_line*4

        msg += trial_msg + last_metric + best_metric + time_msg

        return msg
