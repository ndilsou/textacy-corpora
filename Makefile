project-root=$(shell git rev-parse --show-toplevel)

.DEFAULT_GOAL := help

#########################################################
# HELP

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

#########################################################
set-pythonpath: ## Fixes missing pythonpath when running pipenv
	echo "PYTHONPATH=${PYTHONPATH}:${PWD}" >> .env

init: set-pythonpath
	pipenv install --dev
	pipenv run spacy download en_core_web_sm

clean:
	rm  -rf build/*

clean-all: clean
	rm -rf dist/*

build:
	pipenv run python setup.py bdist_wheel

watch-test: ## Watch the test suite
	pipenv run pytest-watch tests/ src/

test: ## Run the test suite
	pipenv run python -m pytest tests/

bump-%:
	pipenv run bumpversion $*

patch: bump-patch

minor: bump-minor

major: bump-major


lab:
	pipenv run jupyter lab --notebook-dir notebooks



