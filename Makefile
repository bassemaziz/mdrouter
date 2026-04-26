SHELL := /bin/bash

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
CONFIG ?= config/providers.json

.PHONY: help venv install setup run test stop clean

help:
	@echo "Available targets:"
	@echo "  make setup   - Create virtual env and install dev dependencies"
	@echo "  make run     - Start mdrouter with config/providers.json"
	@echo "  make test    - Run test suite"
	@echo "  make clean   - Remove virtual env and pytest cache"

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -e ".[dev]"

setup: install

run:
	$(PYTHON) -m mdrouter --config $(CONFIG)

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf $(VENV) .pytest_cache
