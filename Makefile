export SHELL := /bin/bash
export MPLBACKEND := Agg

test:

	pytest

coverage:

	pytest --cov=lyman --cov-config=.coveragerc lyman

lint:

	flake8 lyman
