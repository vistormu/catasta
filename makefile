.ONESHELL:
SHELL := /bin/zsh

version = 0.5.2
PYTHON ?= python3

test:
	@if [[ -z "$(example)" ]]; then \
		echo "usage: make test example=01"; \
		exit 1; \
	fi; \
	setopt localoptions null_glob; \
	matches=(examples/$(example)-*.py); \
	if (( $${#matches} == 0 )); then \
		echo "no example found for prefix: $(example)"; \
		exit 1; \
	fi; \
	if (( $${#matches} > 1 )); then \
		echo "multiple examples match prefix $(example):"; \
		printf '  %s\n' "$${matches[@]}"; \
		echo "use a more specific prefix, e.g. example=19"; \
		exit 1; \
	fi; \
	script="$${matches[1]##examples/}"; \
	cd examples; \
	PYTHONPATH=.. ../.venv-macos/bin/python "$$script"

unit:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -q

smoke:
	$(PYTHON) -m unittest tests.examples_smoke_test -q

package-check:
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

install-smoke:
	$(PYTHON) scripts/ci/install_smoke_test.py

ci: unit smoke package-check install-smoke

# release guide: docs/release_process.md
tag:
	@if git rev-parse -q --verify "refs/tags/v$(version)" >/dev/null; then \
		echo "local tag v$(version) already exists"; \
		exit 1; \
	fi
	git tag -a v$(version) -m "v$(version)"
	git push origin v$(version)
	@echo "tag v$(version) pushed. wait for github release-check to pass before upload."

upload: package-check
	@if ! git rev-parse -q --verify "refs/tags/v$(version)" >/dev/null; then \
		echo "missing local tag v$(version). run 'make tag' first."; \
		exit 1; \
	fi
	$(PYTHON) -m twine upload dist/*
	@echo "pypi upload complete for v$(version)."

release: package-check
	@if ! git ls-remote --tags origin "refs/tags/v$(version)" | grep -q "refs/tags/v$(version)"; then \
		echo "missing remote tag v$(version). run 'make tag' first."; \
		exit 1; \
	fi
	@if gh release view v$(version) >/dev/null 2>&1; then \
		gh release upload v$(version) dist/* --clobber; \
	else \
		gh release create v$(version) dist/* --verify-tag; \
	fi
	@echo "github release is ready for v$(version)."


.PHONY: test unit smoke package-check install-smoke ci tag upload release
