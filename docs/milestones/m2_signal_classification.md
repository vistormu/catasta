# Milestone 2: Signal Classification

Implemented support for tabular signal classification with `task="signal_classification"`.

## Dataset contract

- Uses split layout: `train/`, `val/`, `test/` with CSV files.
- Requires `input_name` and `output_name`.
- Inputs are loaded as numeric tensors.
- Labels are mapped to class indices from the train split.
- Validation/test labels must exist in the train label set; unknown labels raise `ValueError`.

## Training and evaluation contract

- `Scaffold` treats `signal_classification` labels as class indices (`torch.long`).
- `Scaffold.evaluate()` reports classification metrics through `EvalInfo`.
- `EvalInfo` now accepts `task="signal_classification"` as a classification task.
