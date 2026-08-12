"""
Inference: the closed-form baseline, the learned residual, and the simulator.

Two prediction paths are exposed deliberately, because they trade accuracy for
latency by four orders of magnitude:

``predict_fast``
    Closed-form solution plus the CueNet residual. Sub-millisecond, endpoints
    only, suitable for search or for a real-time aiming aid.
``predict``
    Full numerical simulation for the whole rack, which is what the desktop UI
    animates, alongside the fast prediction and the gap between them.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from cueai.ml.model import CueNet
from cueai.physics.analytic import predict_endpoint
from cueai.physics.ball import Ball
from cueai.physics.constants import ShotParams, TableParams
from cueai.physics.simulator import FEATURE_NAMES, Simulator, shot_feature_vector


class TrajectoryPredictor:
    """Loads whichever CueNet artefacts are present and serves predictions."""

    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)
        self.sim = Simulator(dt=0.001, max_time=14.0, collision_passes=20)
        self.net: CueNet | None = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.ort_session = None
        self._load()

    # ------------------------------------------------------------------ loading

    def _load(self) -> None:
        checkpoint = self.model_dir / "cuenet.pt"
        onnx_path = self.model_dir / "cuenet.onnx"
        if checkpoint.exists():
            data = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.net = CueNet(in_dim=int(data["in_dim"]))
            self.net.load_state_dict(data["state_dict"])
            self.net.eval()
            self.mean = np.asarray(data["scaler_mean"], dtype=np.float32)
            self.scale = np.asarray(data["scaler_scale"], dtype=np.float32)
        if onnx_path.exists():
            try:
                import onnxruntime as ort

                self.ort_session = ort.InferenceSession(
                    str(onnx_path), providers=["CPUExecutionProvider"]
                )
            except Exception:  # pragma: no cover - optional runtime
                self.ort_session = None

    @property
    def ready(self) -> bool:
        """True when a trained residual model is available."""
        return self.net is not None or self.ort_session is not None

    @property
    def backend(self) -> str:
        if self.ort_session is not None:
            return "onnx"
        return "torch" if self.net is not None else "none"

    # --------------------------------------------------------------- prediction

    @staticmethod
    def feature_vector(
        shot: ShotParams,
        cue_pos: tuple[float, float] | np.ndarray,
        obj_pos: tuple[float, float] | np.ndarray | None,
        table: TableParams,
    ) -> np.ndarray:
        cue = np.asarray(cue_pos, dtype=np.float64)
        obj = np.asarray(obj_pos, dtype=np.float64) if obj_pos is not None else None
        return shot_feature_vector(shot, cue, obj, table)

    def _standardise(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            return features.astype(np.float32)
        return ((features - self.mean) / np.clip(self.scale, 1e-8, None)).astype(np.float32)

    def residual_batch(self, features: np.ndarray) -> np.ndarray:
        """Endpoint corrections for a batch of raw feature rows, shape (N, 4)."""
        standardised = self._standardise(np.atleast_2d(features))
        if self.ort_session is not None:
            return np.asarray(
                self.ort_session.run(None, {"features": standardised})[0], dtype=np.float64
            )
        if self.net is not None:
            with torch.no_grad():
                return self.net(torch.from_numpy(standardised)).numpy().astype(np.float64)
        return np.zeros((len(standardised), 4), dtype=np.float64)

    def predict_fast(
        self,
        shot: ShotParams,
        cue_pos: tuple[float, float],
        obj_pos: tuple[float, float] | None = None,
        table: TableParams | None = None,
        use_ml: bool = True,
    ) -> dict:
        """Closed-form endpoints plus the learned residual. No integration."""
        table = table or self.sim.table
        baseline_cue = predict_endpoint(shot, cue_pos, table, radius=self.sim.ball_params.radius)
        baseline_obj = np.asarray(obj_pos, dtype=np.float64) if obj_pos is not None else None
        baseline = np.concatenate(
            [baseline_cue, baseline_obj if baseline_obj is not None else np.zeros(2)]
        )

        features = self.feature_vector(shot, cue_pos, obj_pos, table)
        residual = (
            self.residual_batch(features)[0] if (use_ml and self.ready) else np.zeros(4)
        )
        corrected = baseline + residual
        return {
            "baseline_endpoints": {"cue": baseline[:2].tolist(), "object": baseline[2:].tolist()},
            "residual": residual.tolist(),
            "endpoints": {"cue": corrected[:2].tolist(), "object": corrected[2:].tolist()},
            "backend": self.backend if use_ml else "closed_form",
        }

    def predict(
        self,
        shot: ShotParams,
        cue_pos: tuple[float, float],
        obj_pos: tuple[float, float] | None = None,
        table: TableParams | None = None,
        use_ml: bool = True,
        full_rack: bool = True,
        seed: int | None = 7,
        balls: list[Ball] | None = None,
    ) -> dict:
        """
        Full simulation plus the fast prediction, so the two can be compared.

        The returned ``agreement`` block is what the UI and the API surface: how
        far the sub-millisecond estimate lands from the simulated outcome.
        """
        if table is not None:
            self.sim.table = table

        sim_start = time.perf_counter()
        result = self.sim.simulate_shot(
            shot,
            cue_pos=cue_pos,
            obj_pos=obj_pos,
            full_rack=full_rack,
            seed=seed,
            balls=balls,
        )
        sim_ms = (time.perf_counter() - sim_start) * 1000

        # With a full rack the 8-ball stands in for "the object ball" so that the
        # single-object-ball model trained on two-ball shots still has a referent.
        reference = result.endpoints.get(8, result.endpoints.get(1, np.zeros(2)))
        reference_pos = (float(reference[0]), float(reference[1]))

        fast_start = time.perf_counter()
        fast = self.predict_fast(
            shot, cue_pos, obj_pos=obj_pos or reference_pos, table=self.sim.table, use_ml=use_ml
        )
        fast_ms = (time.perf_counter() - fast_start) * 1000

        simulated_cue = np.asarray(result.endpoints.get(0, np.zeros(2)), dtype=np.float64)
        cue_gap = float(np.linalg.norm(simulated_cue - np.asarray(fast["endpoints"]["cue"])))

        return {
            "features": dict(
                zip(
                    FEATURE_NAMES,
                    self.feature_vector(shot, cue_pos, reference_pos, self.sim.table),
                )
            ),
            "simulated_endpoints": {
                "cue": simulated_cue.tolist(),
                "object": list(reference_pos),
            },
            "fast_prediction": fast,
            "agreement": {"cue_gap_m": cue_gap},
            "timing_ms": {"simulator": sim_ms, "fast": fast_ms},
            "endpoints": {str(k): v.tolist() for k, v in result.endpoints.items()},
            "trajectory": {str(k): v.tolist() for k, v in result.trajectories.items()},
            "ball_meta": {str(k): v for k, v in result.ball_meta.items()},
            "pocketed": {str(k): bool(v) for k, v in result.pocketed.items()},
            "collisions": result.collision_events,
            "cushions": result.cushion_events,
            "times": result.times.tolist(),
            "ml_loaded": self.ready,
        }
