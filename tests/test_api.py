"""Contract tests for the FastAPI surface, driven through the ASGI app directly."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cueai.api.main import app

SOFT_SHOT = {"speed": 1.5, "angle_deg": 5.0, "full_rack": False, "cue_x": 0.6, "cue_y": 0.6}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health_reports_model_state(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["backend"] in {"none", "torch", "onnx"}
    assert isinstance(body["ml_loaded"], bool)


def test_table_geometry_is_a_nine_foot_table(client: TestClient) -> None:
    body = client.get("/table").json()
    assert body["length"] == pytest.approx(2.54)
    assert body["width"] == pytest.approx(1.27)
    assert len(body["pockets"]) == 6


def test_rack_is_a_legal_eight_ball_rack(client: TestClient) -> None:
    balls = client.get("/rack", params={"seed": 7}).json()["balls"]
    assert len(balls) == 16
    assert sorted(b["number"] for b in balls) == list(range(16))
    eight = next(b for b in balls if b["number"] == 8)
    assert eight["suit"] == "eight"


def test_fast_prediction_stays_on_the_table(client: TestClient) -> None:
    body = client.post("/predict/fast", json=SOFT_SHOT).json()
    cue = body["endpoints"]["cue"]
    assert 0.0 <= cue[0] <= 2.54
    assert 0.0 <= cue[1] <= 1.27
    assert body["timing_ms"] < 50


def test_fast_prediction_is_reproducible(client: TestClient) -> None:
    first = client.post("/predict/fast", json=SOFT_SHOT).json()
    second = client.post("/predict/fast", json=SOFT_SHOT).json()
    assert first["endpoints"] == second["endpoints"]


def test_simulation_returns_trajectories_and_timings(client: TestClient) -> None:
    body = client.post("/predict", json={**SOFT_SHOT, "obj_x": 1.4, "obj_y": 0.7}).json()
    cue_path = np.asarray(body["trajectory"]["0"])
    assert cue_path.shape[1] == 2
    assert len(cue_path) > 10
    assert body["timing_ms"]["simulator"] > body["timing_ms"]["fast"]


def test_rejects_out_of_range_shots(client: TestClient) -> None:
    assert client.post("/predict/fast", json={**SOFT_SHOT, "speed": 99.0}).status_code == 422
    assert client.post("/predict/fast", json={**SOFT_SHOT, "english_x": 3.0}).status_code == 422
