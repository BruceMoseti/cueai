"""Inference helpers: physics + ONNX / Torch residual fusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cueai.ml.model import CueNet
from cueai.physics.ball import Ball
from cueai.physics.constants import ShotParams, TableParams
from cueai.physics.simulator import FEATURE_NAMES, Simulator, shot_feature_vector


class TrajectoryPredictor:
    def __init__(self, model_dir: str | Path = "models"):
        self.model_dir = Path(model_dir)
        self.sim = Simulator(dt=0.001, max_time=14.0, collision_passes=20)
        self.net: CueNet | None = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.ort_session = None
        self._load()

    def _load(self) -> None:
        ckpt = self.model_dir / "cuenet.pt"
        onnx = self.model_dir / "cuenet.onnx"
        if ckpt.exists():
            data = torch.load(ckpt, map_location="cpu", weights_only=False)
            self.net = CueNet(in_dim=int(data["in_dim"]))
            self.net.load_state_dict(data["state_dict"])
            self.net.eval()
            self.mean = np.asarray(data["scaler_mean"], dtype=np.float32)
            self.scale = np.asarray(data["scaler_scale"], dtype=np.float32)
        if onnx.exists():
            try:
                import onnxruntime as ort

                self.ort_session = ort.InferenceSession(
                    str(onnx), providers=["CPUExecutionProvider"]
                )
            except Exception:
                self.ort_session = None

    def _scale(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            return x.astype(np.float32)
        return ((x - self.mean) / np.clip(self.scale, 1e-8, None)).astype(np.float32)

    def _residual(self, x: np.ndarray) -> np.ndarray:
        xs = self._scale(x)[None, :]
        if self.ort_session is not None:
            out = self.ort_session.run(None, {"features": xs})[0]
            return np.asarray(out[0], dtype=np.float64)
        if self.net is not None:
            with torch.no_grad():
                out = self.net(torch.from_numpy(xs)).numpy()[0]
            return out.astype(np.float64)
        return np.zeros(4, dtype=np.float64)

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
        if table is not None:
            self.sim.table = table
        result = self.sim.simulate_shot(
            shot,
            cue_pos=cue_pos,
            obj_pos=obj_pos,
            full_rack=full_rack,
            seed=seed,
            balls=balls,
        )
        cue_p = np.array(cue_pos, dtype=np.float64)
        # Use 8-ball as reference object for ML residual (legacy head)
        eight = result.endpoints.get(8, result.endpoints.get(1, np.zeros(2)))
        obj_p = np.asarray(eight, dtype=np.float64)
        feats = shot_feature_vector(shot, cue_p, obj_p, self.sim.table)
        phys = np.array(
            [
                result.endpoints.get(0, np.zeros(2))[0],
                result.endpoints.get(0, np.zeros(2))[1],
                float(obj_p[0]),
                float(obj_p[1]),
            ]
        )
        residual = self._residual(feats) if use_ml else np.zeros(4)
        corrected = phys + residual
        return {
            "features": {k: float(v) for k, v in zip(FEATURE_NAMES, feats)},
            "physics_endpoints": {
                "cue": phys[:2].tolist(),
                "object": phys[2:].tolist(),
            },
            "ml_residual": residual.tolist(),
            "fused_endpoints": {
                "cue": corrected[:2].tolist(),
                "object": corrected[2:].tolist(),
            },
            "endpoints": {str(k): v.tolist() for k, v in result.endpoints.items()},
            "trajectory": {
                str(k): v.tolist() for k, v in result.trajectories.items()
            },
            "ball_meta": {str(k): v for k, v in result.ball_meta.items()},
            "pocketed": {str(k): bool(v) for k, v in result.pocketed.items()},
            "collisions": result.collision_events,
            "times": result.times.tolist(),
            "ml_loaded": self.net is not None or self.ort_session is not None,
        }
