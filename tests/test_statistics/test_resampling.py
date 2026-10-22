"""Tests T17.7 — rééchantillonnage."""

import numpy as np
import pandas as pd
import pytest

from app.errors import InvalidMethodInputError
from app.statistics.resampling import (
    run_bootstrap,
    run_permutation_test,
    simulate_power,
)


def test_bootstrap_mean_reproducible():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    r1 = run_bootstrap(df, "mean", ["x"], n_resamples=500, random_state=42)
    r2 = run_bootstrap(df, "mean", ["x"], n_resamples=500, random_state=42)
    assert r1.estimates["statistic"] == r2.estimates["statistic"]
    lo, hi = r1.confidence_intervals["statistic"]
    assert lo <= r1.estimates["statistic"] <= hi


def test_bootstrap_rejects_unknown_statistic():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(InvalidMethodInputError):
        run_bootstrap(df, "variance", ["x"])


def test_permutation_test():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "outcome": np.concatenate([rng.normal(0, 1, 25), rng.normal(0.5, 1, 25)]),
        "group": ["A"] * 25 + ["B"] * 25,
    })
    result = run_permutation_test(df, "outcome", "group", random_state=1, n_resamples=2000)
    assert 0 <= result.p_values["overall"] <= 1


def test_simulate_power():
    result = simulate_power({"effect_size": 0.8, "n_per_group": 40}, n_simulations=500, random_state=0)
    assert result.estimates["power"] > 0.5
