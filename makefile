.ONESHELL:
SHELL := /bin/zsh

test:
	@cp -r catasta .venv-test/lib/python3.13/site-packages

.PHONY: test
