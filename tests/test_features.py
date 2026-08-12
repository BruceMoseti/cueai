"""
Guards against train/serve skew.

A residual model is only as good as the assumption that features are built the
same way at training time and at request time. These tests pin that down, since
a silent mismatch would show up as a quietly worse model rather than an error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cueai.ml import features as feat
from cueai.ml.dataset import simulate_sample
from cueai.physics.constants import ShotParams, TableParams


def test_feature_names_match_vector_length() -> None:
    vector = feat.build_features(
        ShotParams(speed=2.0, angle=0.3, english_x=0.2, english_y=-0.1),
        (0.6, 0.5),
        (1.4, 0.7),
        TableParams(),
    )
    assert vector.shape == (len(feat.FEATURE_NAMES),)
    assert np.isfinite(vector).all()


def test_serving_path_matches_training_path() -> None:
    """
    The dataset stores raw parameters and rebuilds features in bulk; serving
    builds them one shot at a time. The two must agree exactly.
    """
    rows = [simulate_sample(index, seed=11) for index in range(4)]
    df = pd.DataFrame(rows)
    bulk = feat.build_feature_frame(df)

    for position, record in enumerate(rows):
        shot = ShotParams(
            speed=record["speed"],
            angle=record["angle"],
            english_x=record["english_x"],
            english_y=record["english_y"],
            cue_elevation=record["cue_elevation"],
        )
        table = TableParams(
            mu_slide=record["mu_slide"],
            mu_roll=record["mu_roll"],
            mu_spin=record["mu_spin"],
            e_cushion=record["e_cushion"],
            friction_noise_amp=record["friction_noise_amp"],
        )
        single = feat.build_features(
            shot, (record["cue_x"], record["cue_y"]), (record["obj_x"], record["obj_y"]), table
        )
        np.testing.assert_allclose(single, bulk[position], rtol=0, atol=0)


def test_baseline_features_agree_with_the_stored_baseline() -> None:
    """The closed-form endpoint in the features is the one used as the residual base."""
    record = simulate_sample(3, seed=11)
    vector = feat.build_features(
        ShotParams(
            speed=record["speed"],
            angle=record["angle"],
            english_x=record["english_x"],
            english_y=record["english_y"],
            cue_elevation=record["cue_elevation"],
        ),
        (record["cue_x"], record["cue_y"]),
        (record["obj_x"], record["obj_y"]),
        TableParams(
            mu_slide=record["mu_slide"],
            mu_roll=record["mu_roll"],
            mu_spin=record["mu_spin"],
            e_cushion=record["e_cushion"],
            friction_noise_amp=record["friction_noise_amp"],
        ),
    )
    named = dict(zip(feat.FEATURE_NAMES, vector))
    assert named["base_cue_x"] == pytest.approx(record["baseline_cue_end_x"])
    assert named["base_cue_y"] == pytest.approx(record["baseline_cue_end_y"])


def test_contact_geometry_flags_a_straight_on_shot() -> None:
    table = TableParams()
    straight = feat.build_features(
        ShotParams(speed=2.0, angle=0.0), (0.6, 0.635), (1.2, 0.635), table
    )
    named = dict(zip(feat.FEATURE_NAMES, straight))
    assert named["will_contact"] == 1.0
    assert named["contact_perp"] == pytest.approx(0.0, abs=1e-9)
    assert named["contact_along"] == pytest.approx(0.6)

    wide = feat.build_features(
        ShotParams(speed=2.0, angle=0.6), (0.6, 0.635), (1.2, 0.635), table
    )
    assert dict(zip(feat.FEATURE_NAMES, wide))["will_contact"] == 0.0
