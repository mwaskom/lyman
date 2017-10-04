export SHELL := /bin/bash

test:

	pytest

coverage:

	pytest --cov=lyman --cov-config=.coveragerc lyman

lint:

	flake8 lyman
