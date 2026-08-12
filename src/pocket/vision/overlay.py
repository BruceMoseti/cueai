"""OpenCV helpers for overlaying predicted trajectories on table imagery."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from pocket.physics.constants import TableParams


def render_trajectory_frame(
    trajectory: dict[str, list],
    table: TableParams | None = None,
    width: int = 1000,
    height: int = 500,
) -> np.ndarray:
    """Render a top-down OpenCV visualization (BGR)."""
    if cv2 is None:
        raise RuntimeError("opencv-python is required")
    table = table or TableParams()
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (40, 120, 60)

    def to_px(x: float, y: float) -> tuple[int, int]:
        px = int(x / table.length * (width - 40) + 20)
        py = int((table.width - y) / table.width * (height - 40) + 20)
        return px, py

    # rails
    cv2.rectangle(img, (20, 20), (width - 20, height - 20), (40, 60, 120), 4)
    for px, py in table.pockets:
        cv2.circle(img, to_px(px, py), 14, (0, 0, 0), -1)

    colors = {"0": (80, 220, 255), "1": (80, 80, 255)}
    for bid, path in trajectory.items():
        color = colors.get(str(bid), (200, 200, 200))
        pts = [to_px(p[0], p[1]) for p in path]
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], color, 2)
        if pts:
            cv2.circle(img, pts[-1], 8, color, -1)
    return img


def save_trajectory_image(path: str, trajectory: dict[str, list]) -> str:
    img = render_trajectory_frame(trajectory)
    cv2.imwrite(path, img)
    return path
