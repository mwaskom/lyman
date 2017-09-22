export SHELL := /bin/bash

test:

	pytest

coverage:

	pytest --cov=lyman lyman

lint:

	flake8 lyman
