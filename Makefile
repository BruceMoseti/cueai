.PHONY: help setup test lint typecheck check train bench figures facts api ui play \
        parity parity-check selfplay browser input capture web clean all
.DEFAULT_GOAL := help

PY ?= python3
NODE ?= node
# Defaults reproduce the published numbers in models/metrics.json.
SAMPLES ?= 20000
EPOCHS ?= 300
GAMES ?= 20
PORT ?= 8123

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

check: lint typecheck test  ## Everything CI runs on the Python side

parity:  ## Re-export reference shots and check the browser physics against them
	$(PY) scripts/export_parity_cases.py
	$(NODE) web/test/parity.mjs --verbose
	$(PY) scripts/site_facts.py

parity-check:  ## Verify the committed reference shots without rewriting them
	$(PY) scripts/export_parity_cases.py --check
	$(NODE) web/test/parity.mjs

selfplay:  ## Play the bot against itself headlessly, exercising every rule
	$(NODE) web/test/selfplay.mjs --games $(GAMES) --difficulty sharp --vs relaxed

browser:  ## Load the page in Chrome and play a game (needs puppeteer-core)
	$(NODE) web/test/browser.mjs --url http://localhost:$(PORT)/index.html --games 2
	$(NODE) web/test/input.mjs --url http://localhost:$(PORT)/index.html

capture:  ## Re-record the screenshots and the clip in the README
	$(NODE) web/test/capture.mjs --url http://localhost:$(PORT)/index.html

web: parity selfplay  ## Every check that does not need a browser

play:  ## Serve the game at http://localhost:$(PORT)
	@echo "CueAI is at http://localhost:$(PORT)"
	@cd web && $(PY) -m http.server $(PORT)

train:  ## Generate data and train the residual model
	$(PY) -m cueai.ml.train --n-samples $(SAMPLES) --epochs $(EPOCHS)

bench:  ## Measure latency and rewrite docs/BENCHMARKS.md
	$(PY) scripts/benchmark.py

figures:  ## Render the README figures into docs/assets
	$(PY) scripts/make_figures.py

facts:  ## Rewrite the numbers the playable page quotes
	$(PY) scripts/site_facts.py

api:  ## Serve the prediction API on :8000
	$(PY) -m uvicorn cueai.api.main:app --reload --port 8000

ui:  ## Launch the desktop table (needs the ui extra)
	$(PY) -m cueai.ui.app

all: train bench figures facts  ## Reproduce every published number and figure

clean:  ## Remove generated artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__
	rm -f models/*.pt models/*.onnx models/*.joblib
