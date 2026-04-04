# release process

this is the canonical publish flow for `catasta`.

## prerequisites

- working tree is clean (or only intended release changes are present).
- you are on the branch/commit you want to release.
- `version` in `makefile` and `pyproject.toml` is already updated.
- optional: set python once if needed:
  - `export PYTHON=.venv-macos/bin/python`

## 1) local quality gate

run:

```sh
make ci PYTHON=${PYTHON:-python3}
```

this runs unit tests, example smoke tests, packaging checks, and install smoke checks.

## 2) create and push tag

run:

```sh
make tag
```

this creates `v$(version)` and pushes it to origin.

## 3) wait for github checks

wait until the github `release-check` workflow for the tag passes.

## 4) publish to pypi

run:

```sh
make upload PYTHON=${PYTHON:-python3}
```

this rebuilds artifacts, validates them (`twine check`), and uploads to pypi.

## 5) create/update github release

run:

```sh
make release PYTHON=${PYTHON:-python3}
```

if the release exists, assets are updated with `--clobber`; otherwise the release is created.

## quick command summary

```sh
make ci PYTHON=.venv-macos/bin/python
make tag
# wait for github release-check
make upload PYTHON=.venv-macos/bin/python
make release PYTHON=.venv-macos/bin/python
```
