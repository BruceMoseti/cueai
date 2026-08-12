"""PyQt6 interactive CueAI — full 16-ball table."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from cueai.ml.infer import TrajectoryPredictor
from cueai.physics.ball import Ball
from cueai.physics.constants import ShotParams, TableParams
from cueai.physics.rack import make_full_rack


class TableCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(980, 520)
        self.table = TableParams()
        self.R = 0.028575
        self.balls: list[Ball] = make_full_rack(self.table, seed=7)
        self.trajectory: dict[str, list] = {}
        self.ball_meta: dict[str, dict] = {}
        self.pocketed: dict[str, bool] = {}
        self.anim_index = 0
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._tick)
        self.aim_angle = 0.0
        self.dragging: int | None = None  # ball id
        self.show_paths = True

    @property
    def cue(self) -> Ball:
        for b in self.balls:
            if b.number == 0:
                return b
        return self.balls[0]

    def reset_rack(self, seed: int | None = 7) -> None:
        cue_y = float(self.cue.pos[1])
        self.balls = make_full_rack(
            self.table,
            cue_pos=(self.table.length * 0.25, cue_y),
            seed=seed,
        )
        self.trajectory = {}
        self.pocketed = {}
        self.anim_index = 0
        self.anim_timer.stop()
        self.update()

    def apply_final_positions(self, result: dict) -> None:
        """Freeze balls at post-shot resting positions for continued play."""
        endpoints = result.get("endpoints", {})
        pocketed = result.get("pocketed", {})
        for b in self.balls:
            key = str(b.id)
            if pocketed.get(key) or pocketed.get(b.id):
                b.pocketed = True
                b.pos = np.array([-1.0, -1.0])
                b.vel[:] = 0
                continue
            ep = endpoints.get(key)
            if ep is not None:
                b.pos = np.array(ep, dtype=np.float64)
                b.vel[:] = 0
                b.omega[:] = 0
                b.pocketed = False

    def world_to_screen(self, x: float, y: float) -> QPointF:
        margin = 44
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin
        sx = margin + (x / self.table.length) * w
        sy = margin + ((self.table.width - y) / self.table.width) * h
        return QPointF(sx, sy)

    def screen_to_world(self, sx: float, sy: float) -> np.ndarray:
        margin = 44
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin
        x = (sx - margin) / w * self.table.length
        y = self.table.width - (sy - margin) / h * self.table.width
        return np.array([x, y], dtype=np.float64)

    def set_result(self, result: dict) -> None:
        self.trajectory = result.get("trajectory", {})
        self.ball_meta = result.get("ball_meta", {})
        self.pocketed = {str(k): bool(v) for k, v in result.get("pocketed", {}).items()}
        self.anim_index = 0
        if self.trajectory:
            self.anim_timer.start(14)

    def _tick(self) -> None:
        cue_path = self.trajectory.get("0", [])
        if not cue_path:
            self.anim_timer.stop()
            return
        self.anim_index += 1
        max_len = max((len(v) for v in self.trajectory.values()), default=0)
        if self.anim_index >= max_len:
            self.anim_timer.stop()
        self.update()

    def _ball_pos_at_frame(self, bid: int) -> np.ndarray | None:
        path = self.trajectory.get(str(bid), [])
        if not path:
            for b in self.balls:
                if b.id == bid:
                    return None if b.pocketed else b.pos
            return None
        if self.pocketed.get(str(bid)) and self.anim_index >= len(path) - 1:
            # finished pocketed
            last = path[-1]
            if last[0] < 0:
                return None
        idx = min(self.anim_index, len(path) - 1)
        pos = path[idx]
        if pos[0] < 0:
            return None
        return np.asarray(pos, dtype=np.float64)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(14, 24, 20))

        tl = self.world_to_screen(0, self.table.width)
        br = self.world_to_screen(self.table.length, 0)
        felt = QRectF(tl, br)
        p.fillRect(felt, QColor(18, 118, 72))
        # Cloth grain
        p.setPen(QPen(QColor(14, 100, 60, 40), 1))
        for i in range(12):
            y = tl.y() + (br.y() - tl.y()) * i / 12
            p.drawLine(QPointF(tl.x(), y), QPointF(br.x(), y))

        p.setPen(QPen(QColor(92, 48, 28), 10))
        p.drawRect(felt)

        # Pockets
        p.setBrush(QBrush(QColor(8, 8, 8)))
        p.setPen(Qt.PenStyle.NoPen)
        pr = self.table.pocket_radius
        for px, py in self.table.pockets:
            c = self.world_to_screen(px, py)
            r = abs(self.world_to_screen(px + pr, py).x() - c.x())
            p.drawEllipse(c, r, r)

        # Head string
        hx = self.table.length * 0.25
        p.setPen(QPen(QColor(255, 255, 255, 35), 1, Qt.PenStyle.DashLine))
        p.drawLine(
            self.world_to_screen(hx, 0.02),
            self.world_to_screen(hx, self.table.width - 0.02),
        )

        # Trails
        if self.show_paths and self.trajectory:
            for bid, path in self.trajectory.items():
                if len(path) < 2:
                    continue
                meta = self.ball_meta.get(bid, {})
                col = meta.get("color", [200, 200, 200])
                pen = QPen(QColor(col[0], col[1], col[2], 140), 2)
                p.setPen(pen)
                n = min(self.anim_index + 1, len(path))
                for i in range(1, n):
                    if path[i][0] < 0 or path[i - 1][0] < 0:
                        continue
                    p.drawLine(
                        self.world_to_screen(*path[i - 1]),
                        self.world_to_screen(*path[i]),
                    )

        br_px = abs(self.world_to_screen(self.R, 0).x() - self.world_to_screen(0, 0).x())

        # Draw balls
        for b in self.balls:
            pos = self._ball_pos_at_frame(b.id)
            if pos is None:
                continue
            self._draw_ball(p, b, pos, br_px)

        # Aim line from cue
        cue = self.cue
        if not cue.pocketed:
            aim_len = 0.35
            tip = cue.pos + aim_len * np.array(
                [math.cos(self.aim_angle), math.sin(self.aim_angle)]
            )
            p.setPen(QPen(QColor(255, 255, 255, 150), 1, Qt.PenStyle.DotLine))
            p.drawLine(self.world_to_screen(*cue.pos), self.world_to_screen(*tip))

        p.setPen(QColor(210, 230, 210))
        p.setFont(QFont("Avenir Next", 12, QFont.Weight.DemiBold))
        p.drawText(18, 24, "CueAI — full rack · spin · throw · multi-collision")

    def _draw_ball(self, p: QPainter, b: Ball, pos: np.ndarray, br_px: float) -> None:
        c = self.world_to_screen(*pos)
        color = (245, 245, 240)
        stripe = False
        if b.identity:
            color = b.identity.color
            stripe = b.identity.is_stripe
        elif str(b.id) in self.ball_meta:
            meta = self.ball_meta[str(b.id)]
            color = tuple(meta.get("color", color))
            stripe = bool(meta.get("stripe"))

        if stripe:
            p.setBrush(QBrush(QColor(245, 245, 240)))
            p.setPen(QPen(QColor(30, 30, 30), 1))
            p.drawEllipse(c, br_px, br_px)
            p.setBrush(QBrush(QColor(*color)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(
                QRectF(c.x() - br_px, c.y() - br_px * 0.45, 2 * br_px, br_px * 0.9)
            )
            p.setPen(QPen(QColor(30, 30, 30), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, br_px, br_px)
        else:
            p.setBrush(QBrush(QColor(*color)))
            p.setPen(QPen(QColor(20, 20, 20), 1))
            p.drawEllipse(c, br_px, br_px)

        # Number badge
        if b.number != 0:
            badge = br_px * 0.45
            p.setBrush(QBrush(QColor(250, 250, 250)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(c, badge, badge)
            p.setPen(QColor(20, 20, 20))
            font = QFont("Avenir Next", max(7, int(br_px * 0.55)))
            font.setBold(True)
            p.setFont(font)
            text = str(b.number)
            p.drawText(
                QRectF(c.x() - badge, c.y() - badge, 2 * badge, 2 * badge),
                int(Qt.AlignmentFlag.AlignCenter),
                text,
            )

    def mousePressEvent(self, event) -> None:
        w = self.screen_to_world(event.position().x(), event.position().y())
        # Prefer cue drag; otherwise aim
        if np.linalg.norm(w - self.cue.pos) < 0.07 and not self.cue.pocketed:
            self.dragging = 0
            return
        for b in self.balls:
            if b.pocketed or b.number == 0:
                continue
            if np.linalg.norm(w - b.pos) < 0.06:
                self.dragging = b.id
                return
        delta = w - self.cue.pos
        self.aim_angle = float(math.atan2(delta[1], delta[0]))
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self.dragging is None:
            return
        w = self.screen_to_world(event.position().x(), event.position().y())
        w[0] = float(np.clip(w[0], self.R, self.table.length - self.R))
        w[1] = float(np.clip(w[1], self.R, self.table.width - self.R))
        for b in self.balls:
            if b.id == self.dragging:
                # Prevent heavy overlap while dragging
                ok = True
                for o in self.balls:
                    if o.id == b.id or o.pocketed:
                        continue
                    if np.linalg.norm(w - o.pos) < 2 * self.R * 0.95:
                        ok = False
                        break
                if ok:
                    b.pos = w
                break
        self.update()

    def mouseReleaseEvent(self, _event) -> None:
        self.dragging = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CueAI — Full Rack Billiards")
        self.resize(1280, 700)
        root = Path(__file__).resolve().parents[3]
        self.predictor = TrajectoryPredictor(model_dir=root / "models")
        self._rack_seed = 7

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.canvas = TableCanvas()
        layout.addWidget(self.canvas, stretch=3)

        panel = QVBoxLayout()
        form = QFormLayout()
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.2, 12)
        self.speed.setValue(4.5)
        self.speed.setSingleStep(0.1)
        self.english_x = QDoubleSpinBox()
        self.english_x.setRange(-1, 1)
        self.english_x.setSingleStep(0.05)
        self.english_y = QDoubleSpinBox()
        self.english_y.setRange(-1, 1)
        self.english_y.setSingleStep(0.05)
        self.elevation = QDoubleSpinBox()
        self.elevation.setRange(0, 25)
        self.elevation.setSuffix("°")
        self.mu = QDoubleSpinBox()
        self.mu.setRange(0.05, 0.4)
        self.mu.setValue(0.2)
        self.mu.setSingleStep(0.01)
        self.noise = QDoubleSpinBox()
        self.noise.setRange(0, 0.08)
        self.noise.setValue(0.025)
        self.noise.setSingleStep(0.005)
        self.use_ml = QCheckBox("ML residual fusion")
        self.use_ml.setChecked(True)
        self.show_paths = QCheckBox("Show trails")
        self.show_paths.setChecked(True)
        self.show_paths.toggled.connect(self._toggle_paths)

        form.addRow("Speed (m/s)", self.speed)
        form.addRow("Sidespin", self.english_x)
        form.addRow("Top / backspin", self.english_y)
        form.addRow("Cue elevation", self.elevation)
        form.addRow("Cloth μ_slide", self.mu)
        form.addRow("Table noise", self.noise)
        panel.addLayout(form)
        panel.addWidget(self.use_ml)
        panel.addWidget(self.show_paths)

        self.angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.angle_slider.setRange(-180, 180)
        self.angle_slider.setValue(0)
        self.angle_slider.valueChanged.connect(self._on_angle)
        panel.addWidget(QLabel("Aim angle (°)"))
        panel.addWidget(self.angle_slider)

        shoot = QPushButton("Break / Shoot")
        shoot.clicked.connect(self.shoot)
        panel.addWidget(shoot)

        rack = QPushButton("Re-rack")
        rack.clicked.connect(self.rerack)
        panel.addWidget(rack)

        self.status = QLabel(
            "Full 16-ball rack. Drag cue or object balls · click felt to aim · Shoot."
        )
        self.status.setWordWrap(True)
        panel.addWidget(self.status)
        panel.addStretch()
        wrap = QWidget()
        wrap.setLayout(panel)
        wrap.setMaximumWidth(290)
        layout.addWidget(wrap)

    def _toggle_paths(self, on: bool) -> None:
        self.canvas.show_paths = on
        self.canvas.update()

    def _on_angle(self, v: int) -> None:
        self.canvas.aim_angle = math.radians(v)
        self.canvas.update()

    def rerack(self) -> None:
        self._rack_seed = int(self._rack_seed) + 1
        self.canvas.reset_rack(seed=self._rack_seed)
        self.status.setText(f"Re-racked (seed {self._rack_seed}).")

    def shoot(self) -> None:
        import math as _math

        shot = ShotParams(
            speed=self.speed.value(),
            angle=self.canvas.aim_angle,
            english_x=self.english_x.value(),
            english_y=self.english_y.value(),
            cue_elevation=_math.radians(self.elevation.value()),
        )
        table = TableParams(
            mu_slide=self.mu.value(),
            friction_noise_amp=self.noise.value(),
        )
        # Continue from the current table state, pocketed markers included
        all_balls = [b.copy() for b in self.canvas.balls]
        result = self.predictor.predict(
            shot,
            cue_pos=(float(self.canvas.cue.pos[0]), float(self.canvas.cue.pos[1])),
            table=table,
            use_ml=self.use_ml.isChecked(),
            full_rack=True,
            balls=all_balls,
        )
        self.canvas.set_result(result)

        def _freeze() -> None:
            self.canvas.apply_final_positions(result)
            self.canvas.trajectory = {}
            self.canvas.update()

        # After animation finishes, freeze positions (approx by timer)
        QTimer.singleShot(max(800, int(len(result.get("times", [])) * 14)), _freeze)

        sunk = [k for k, v in result["pocketed"].items() if v and k != "0"]
        cue_scratch = bool(result["pocketed"].get("0"))
        ml = "ON" if result["ml_loaded"] and self.use_ml.isChecked() else "OFF"
        self.status.setText(
            f"ML {ml} · collisions≈{result.get('collisions', 0)} · "
            f"pocketed {len(sunk)} {sunk[:6]}{'…' if len(sunk) > 6 else ''}"
            + (" · SCRATCH" if cue_scratch else "")
        )


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
