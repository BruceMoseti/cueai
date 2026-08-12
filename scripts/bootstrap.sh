#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]" 2>/dev/null || pip install -r requirements.txt && pip install -e .
python -m pocket.ml.train --n-samples 800 --epochs 15
python -m pytest tests/ -q
echo "Pocket Physics setup complete."
echo "  API:  uvicorn pocket.api.main:app --reload --port 8000"
echo "  UI:   python -m pocket.ui.app"
