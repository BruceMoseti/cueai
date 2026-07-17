"""CueNet: physics-informed residual regressor for trajectory endpoints."""

from __future__ import annotations

import torch
import torch.nn as nn


class CueNet(nn.Module):
    """
    Predicts residual corrections on top of classical physics endpoints.

    Input: shot + table features (18)
    Output: Δ(cue_x, cue_y, obj_x, obj_y)  — add to physics baseline
    """

    def __init__(self, in_dim: int = 18, hidden: int = 128, out_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, out_dim),
        )
        # Start near identity residual (physics-only)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridPredictor(nn.Module):
    """Physics baseline features concatenated; residual applied to last 4 phys cols optional."""

    def __init__(self, in_dim: int = 18):
        super().__init__()
        self.residual = CueNet(in_dim=in_dim)

    def forward(self, x: torch.Tensor, phys_endpoints: torch.Tensor) -> torch.Tensor:
        """
        x: (B, F) features
        phys_endpoints: (B, 4) classical simulator endpoints
        returns corrected endpoints (B, 4)
        """
        return phys_endpoints + self.residual(x)
