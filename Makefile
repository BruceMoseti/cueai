.PHONY: help setup test lint typecheck check train bench figures api ui clean all
.DEFAULT_GOAL := help

PY ?= python3
SAMPLES ?= 20000
EPOCHS ?= 200

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the package with dev extras
	$(PY) -m pip install -e ".[dev]"

test:  ## Run the test suite
	$(PY) -m pytest tests/ -q

lint:  ## Check style
	$(PY) -m ruff check src tests scripts

typecheck:  ## Run static type checks
	$(PY) -m mypy

check: lint typecheck test  ## Everything CI runs

train:  ## Generate data and train the residual model
	$(PY) -m cueai.ml.train --n-samples $(SAMPLES) --epochs $(EPOCHS)

bench:  ## Measure latency and rewrite docs/BENCHMARKS.md
	$(PY) scripts/benchmark.py

figures:  ## Render the README figures into docs/assets
	$(PY) scripts/make_figures.py

api:  ## Serve the prediction API on :8000
	$(PY) -m uvicorn cueai.api.main:app --reload --port 8000

ui:  ## Launch the desktop table (needs the ui extra)
	$(PY) -m cueai.ui.app

all: train bench figures  ## Reproduce every published number and figure

clean:  ## Remove generated artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__
	rm -f models/*.pt models/*.onnx models/*.joblib
