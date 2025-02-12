.ONESHELL:
SHELL := /bin/zsh

version = 0.4.1

test:
	@rm -rf .venv-test/lib/python3.13/site-packages/catasta
	@cp -r catasta .venv-test/lib/python3.13/site-packages

upload:
	python -m build
	python -m twine upload dist/*
	git tag -a v$(version) -m "v$(version)"
	git push origin v$(version)
	gh release create v$(version) dist/*


.PHONY: test
