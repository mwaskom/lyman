export SHELL := /bin/bash

test:

	nosetests -v

coverage:

	nosetests --cover-erase --with-coverage --cover-html --cover-package lyman

lint:

	pyflakes -x W lyman
	pep8 --exclude lyman/workflows/archive lyman
