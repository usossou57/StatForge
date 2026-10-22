"""Tests T17.5 — puissance."""

import pytest

from app.statistics.power import (
    compute_detectable_effect,
    compute_power,
    compute_sample_size,
)


def test_compute_power_t_increases_with_n():
    low = compute_power("t", effect_size=0.8, nobs=15).estimates["power"]
    high = compute_power("t", effect_size=0.8, nobs=80).estimates["power"]
    assert high > low


def test_compute_sample_size_returns_positive_n():
    result = compute_sample_size("t", effect_size=0.5, power=0.8)
    assert result.estimates["n_required"] >= 10


def test_compute_detectable_effect():
    result = compute_detectable_effect("t", nobs=50, power=0.8)
    assert result.estimates["effect_size"] > 0


def test_unknown_test_raises():
    from app.errors import InvalidMethodInputError

    with pytest.raises(InvalidMethodInputError):
        compute_power("unknown", effect_size=0.5, nobs=30)
