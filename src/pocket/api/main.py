"""FastAPI backend for Pocket Physics full-rack simulation + prediction."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pocket.ml.dataset import MAX_TIP_OFFSET
from pocket.ml.infer import TrajectoryPredictor
from pocket.physics.constants import ShotParams, TableParams
from pocket.physics.rack import make_full_rack

API_VERSION = "0.3.0"

app = FastAPI(
    title="Pocket Physics API",
    description=(
        "Physics-informed billiards prediction. /predict runs the full numerical "
        "simulation; /predict/fast returns the closed-form plus learned-residual "
        "estimate in under a millisecond."
    ),
    version=API_VERSION,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_MODEL_DIR = Path(__file__).resolve().parents[3] / "models"
_predictor = TrajectoryPredictor(model_dir=_MODEL_DIR)


class ShotRequest(BaseModel):
    speed: float = Field(4.5, ge=0.1, le=12)
    angle_deg: float = Field(0.0, description="Launch angle in degrees")
    # Tip offsets are bounded by the miscue limit, which is also where the
    # training distribution stops: a request outside it would be extrapolation.
    english_x: float = Field(
        0.0, ge=-MAX_TIP_OFFSET, le=MAX_TIP_OFFSET, description="Sidespin tip offset, fraction of R"
    )
    english_y: float = Field(
        0.0, ge=-MAX_TIP_OFFSET, le=MAX_TIP_OFFSET, description="Top/backspin tip offset"
    )
    cue_elevation_deg: float = 0.0
    cue_x: float = 0.635
    cue_y: float = 0.635
    obj_x: float | None = None
    obj_y: float | None = None
    mu_slide: float = 0.2
    friction_noise_amp: float = 0.025
    use_ml: bool = True
    full_rack: bool = True
    rack_seed: int = 7


class HealthResponse(BaseModel):
    status: str
    ml_loaded: bool
    backend: str
    version: str


def _to_physics(req: ShotRequest) -> tuple[ShotParams, TableParams, tuple[float, float] | None]:
    shot = ShotParams(
        speed=req.speed,
        angle=float(np.deg2rad(req.angle_deg)),
        english_x=req.english_x,
        english_y=req.english_y,
        cue_elevation=float(np.deg2rad(req.cue_elevation_deg)),
    )
    table = TableParams(mu_slide=req.mu_slide, friction_noise_amp=req.friction_noise_amp)
    obj = (req.obj_x, req.obj_y) if req.obj_x is not None and req.obj_y is not None else None
    return shot, table, obj


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        ml_loaded=_predictor.ready,
        backend=_predictor.backend,
        version=API_VERSION,
    )


@app.post("/predict")
def predict(req: ShotRequest) -> dict:
    """Full numerical simulation, with the fast estimate alongside for comparison."""
    shot, table, obj = _to_physics(req)
    return _predictor.predict(
        shot,
        cue_pos=(req.cue_x, req.cue_y),
        obj_pos=obj,
        table=table,
        use_ml=req.use_ml,
        full_rack=req.full_rack,
        seed=req.rack_seed,
    )


@app.post("/predict/fast")
def predict_fast(req: ShotRequest) -> dict:
    """Closed-form plus learned residual. No integration, sub-millisecond."""
    shot, table, obj = _to_physics(req)
    started = time.perf_counter()
    result = _predictor.predict_fast(
        shot, cue_pos=(req.cue_x, req.cue_y), obj_pos=obj, table=table, use_ml=req.use_ml
    )
    result["timing_ms"] = (time.perf_counter() - started) * 1000
    return result


@app.get("/rack")
def rack(seed: int = 7) -> dict:
    balls = make_full_rack(seed=seed)
    return {
        "balls": [
            {
                "id": b.id,
                "number": b.number,
                "suit": b.identity.suit if b.identity else None,
                "color": list(b.identity.color) if b.identity else None,
                "stripe": bool(b.identity and b.identity.is_stripe),
                "pos": b.pos.tolist(),
            }
            for b in balls
        ]
    }


@app.get("/table")
def table_info() -> dict:
    t = TableParams()
    return {
        "length": t.length,
        "width": t.width,
        "pockets": t.pockets,
        "pocket_radius": t.pocket_radius,
    }


def run() -> None:
    import uvicorn

    uvicorn.run("pocket.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
