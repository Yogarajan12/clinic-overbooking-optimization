.PHONY: help install data features models optimise simulate all test lint clean

PYTHON ?= python

help:
	@echo "install   install the package and pipeline dependencies"
	@echo "features  stage 1: clean the Kaggle data, build features, split chronologically"
	@echo "models    stage 2: train, calibrate and audit the prediction models"
	@echo "optimise  stage 3: cost model, convexity check, constrained optimisation"
	@echo "simulate  stage 4: Monte Carlo policy comparison, sensitivity, stress tests"
	@echo "all       run every stage in order"
	@echo "test      run the unit tests"
	@echo "lint      run ruff over the library and tests"

install:
	$(PYTHON) -m pip install -e ".[pipeline,dev]"

features:
	$(PYTHON) pipeline/01_preprocess_and_features.py

models:
	$(PYTHON) pipeline/02_train_and_audit_models.py

optimise:
	$(PYTHON) pipeline/03_optimise_overbooking.py

simulate:
	$(PYTHON) pipeline/04_simulate_and_validate.py

all: features models optimise simulate

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

clean:
	rm -rf results/figures/* results/tables/* results/models/* results/stage3/*
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
