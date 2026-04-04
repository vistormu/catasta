# Milestone 1: API Contracts

This milestone formalizes contracts for upcoming features so behavior is explicit and testable.

## 1) `Scaffold.train` scheduler contract

New argument:

- `scheduler`: `None | LRScheduler | ReduceLROnPlateau | tuple[scheduler, step_on]`

Rules:

- `None`: no scheduler updates.
- Scheduler instance: stepped every epoch.
- Tuple form: `(scheduler, step_on)` where `step_on` is:
  - `"epoch"`: call `scheduler.step()` each epoch.
  - `"val_loss"`: requires `ReduceLROnPlateau`; calls `scheduler.step(val_loss)`.
- The scheduler must be built from the same optimizer instance used in `Scaffold.train`.

## 2) `Scaffold.train` early stopping contract

New argument:

- `early_stopping`: `None | tuple`

Rules:

- Trend mode: `("trend", alpha, patience)`
  - `alpha` in `[0, 1)`.
  - `patience` is an integer `>= 1`.
  - Stop condition: stop after `patience` consecutive epochs where the smoothed validation-loss derivative is positive.
- Plateau mode: `("plateau", patience, min_delta)`
  - `patience` is an integer `>= 1`.
  - `min_delta` is a float `>= 0`.
  - Stop condition: stop after `patience` epochs without improvement larger than `min_delta`.

Compatibility:
- No compatibility aliases are supported. The policy must be explicitly tagged as `"trend"` or `"plateau"`.

## 3) Signal classification scope (contract-only in M1)

The task string `"signal_classification"` is reserved for the new task family.
Implementation of data/model/evaluation flow is deferred to Milestone 2.
