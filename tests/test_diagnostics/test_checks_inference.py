"""Tests T17.8 — diagnostics d'inférence."""

import numpy as np

from app.diagnostics.checks_inference import (
    check_equal_variances,
    check_expected_counts,
    check_group_sizes,
)
from app.diagnostics.models import Severity


def test_unequal_variances_detected():
    g1 = np.random.default_rng(0).normal(0, 1, 30)
    g2 = np.random.default_rng(1).normal(0, 5, 30)
    findings = check_equal_variances([g1, g2], alpha=0.05)
    assert any(f.code == "UNEQUAL_VARIANCES" for f in findings)


def test_small_group_size():
    findings = check_group_sizes([np.array([1.0]), np.array([2.0, 3.0])], min_size=5)
    assert any(f.code == "SMALL_GROUP_SIZE" for f in findings)
    assert findings[0].severity == Severity.WARNING


def test_low_expected_counts():
    table = np.array([[1, 2], [3, 4]])
    findings = check_expected_counts(table, threshold=5)
    assert any(f.code == "LOW_EXPECTED_COUNTS" for f in findings)


def test_clean_data_no_false_positive_variances():
    rng = np.random.default_rng(3)
    groups = [rng.normal(0, 1, 40), rng.normal(0, 1, 40)]
    findings = check_equal_variances(groups, alpha=0.01)
    assert not any(f.code == "UNEQUAL_VARIANCES" for f in findings)
