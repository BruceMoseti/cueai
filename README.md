# CueAI

Physics-informed AI simulation for billiards that combines classical mechanics with
regression models to predict realistic ball trajectories from spin, launch angle,
velocity, cushion interactions, and table surface variations.

| Layer | Tech |
|-------|------|
| Physics core | C++ (optional) + Python NumPy simulator |
| ML | PyTorch → ONNX, Scikit-learn baselines |
| Data | Pandas, NumPy synthetic shot datasets |
| Vision | OpenCV table / cue overlay helpers |
| Backend | FastAPI |
| Frontend | PyQt6 interactive table |

## Quick start

```bash
cd ~/Projects/cueai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate data + train ML model
python -m cueai.ml.train --n-samples 4000 --epochs 40

# API server
uvicorn cueai.api.main:app --reload --port 8000

# Desktop UI
python -m cueai.ui.app
```

## Architecture

```
Shot params (V, θ, spin ω, table μ)
        │
        ▼
┌───────────────────┐     residual correction
│ Physics simulator │ ──► ┌─────────────────┐
│ (cloth, collide,  │     │ CueNet (PyTorch)│──► ONNX
│  cushions)        │     └─────────────────┘
└───────────────────┘              │
        │                          ▼
        └──────────► fused trajectory ──► API / PyQt UI
```

## Resume bullets (use the AI-focused version)

> Developed a physics-informed AI simulation for billiards that combines classical
> mechanics with regression models to predict realistic ball trajectories based on
> spin, launch angle, velocity, cushion interactions, and table surface variations.

## License

MIT
