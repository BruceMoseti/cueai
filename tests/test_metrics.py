"""
Tests for the reported evaluation metrics.

These functions produce the numbers the README quotes, so a silent error here
would misstate the results rather than fail a build. The gate direction in
particular is easy to invert without anything looking wrong.
"""

from __future__ import annotations

import numpy as np

from pocket.ml.train import endpoint_errors, risk_coverage


def test_endpoint_errors_measures_distance_not_axes() -> None:
    """A 3-4-5 offset on both balls is a 5 mm error, not a 3.5 mm mean of axes."""
    truth = np.zeros((1, 4))
    pred = np.array([[0.003, 0.004, 0.003, 0.004]])
    errors = endpoint_errors(pred, truth)
    assert errors["euclidean_mm"] == 5.0
    assert errors["cue_mm"] == 5.0
    assert errors["obj_mm"] == 5.0


def test_risk_coverage_gates_on_the_expected_cushion_count() -> None:
    """
    Tighter gates must answer fewer shots and, when the gate carries signal,
    answer them more accurately.
    """
    expected = np.repeat([0, 1, 2, 3], 10).astype(float)
    truth = np.zeros((40, 4))
    # Error grows with the expected cushion count, which is the pattern the
    # real gate exploits.
    pred = np.tile((expected * 0.1)[:, None], (1, 4))

    rows = risk_coverage({"model": pred}, truth, expected)
    labels = [row["expected_cushions_at_most"] for row in rows]
    assert labels == ["0", "1", "2", "3", "all"]

    coverage = [row["coverage_pct"] for row in rows]
    assert coverage == sorted(coverage)
    assert coverage[-1] == 100.0

    errors = [row["model"] for row in rows]
    assert errors == sorted(errors)
    assert errors[0] == 0.0


def test_risk_coverage_skips_thresholds_with_no_shots() -> None:
    expected = np.full(5, 3.0)
    truth = np.zeros((5, 4))
    rows = risk_coverage({"model": truth}, truth, expected)
    assert [row["expected_cushions_at_most"] for row in rows] == ["3", "all"]
