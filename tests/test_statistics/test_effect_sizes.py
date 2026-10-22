"""Tests T17.4 — tailles d'effet."""

import numpy as np

from app.statistics.effect_sizes import cohens_d, cramers_v, hedges_g, odds_ratio


def test_cohens_d_known_difference():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([3.0, 4.0, 5.0, 6.0])
    out = cohens_d(x, y)
    assert out["value"] is not None
    assert out["value"] < 0
    assert out["ci_low"] is not None and out["ci_high"] is not None


def test_hedges_g_smaller_than_cohens_d():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    d = cohens_d(x, y)["value"]
    g = hedges_g(x, y)["value"]
    assert d is not None and g is not None
    assert abs(g) <= abs(d) + 1e-9


def test_odds_ratio_symmetry():
    out = odds_ratio(10, 5, 2, 20)
    assert out["value"] == 20.0
    assert out["ci_low"] < out["value"] < out["ci_high"]


def test_cramers_v():
    out = cramers_v(chi2=10.0, n=100, min_dim=3)
    assert out["value"] is not None
    assert 0 <= out["value"] <= 1
