project-root=$(shell git rev-parse --show-toplevel)

.DEFAULT_GOAL := help

#########################################################
# HELP

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

#########################################################
clean:
	rm  -rf build/*

clean-all: clean
	rm -rf dist/*

FORCE: 

build: FORCE
	pipenv run python setup.py bdist_wheel

bump-%:
	pipenv run bumpversion $*

patch: bump-patch

minor: bump-minor

major: bump-major

init:
	mkdir -p artifacts data models
	pipenv install --dev

lab:
	pipenv run jupyter lab --notebook-dir notebooks

set-pythonpath: ## Fixes missing pythonpath when running pipenv
	echo "PYTHONPATH=${PYTHONPATH}:${PWD}" >> .env


