from optuna.samplers import (
    BaseSampler,
    RandomSampler,
    TPESampler,
    GPSampler,
)


def get_sampler(sampler: str) -> BaseSampler:
    match sampler.lower():
        case "random" | "rs":
            return RandomSampler()
        case "tpe":
            return TPESampler()
        case "gp" | "bogp":
            return GPSampler()
        case _:
            raise ValueError("Invalid sampler")
