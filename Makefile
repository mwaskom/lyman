export SHELL := /bin/bash

test:

	py.test

coverage:

	nosetests --cover-erase --with-coverage --cover-html --cover-package lyman

lint:

	pyflakes -x W lyman
	pep8 --exclude lyman/workflows/archive lyman
