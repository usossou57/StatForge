"""Tests T17.6 — tests multiples et équivalence."""

import pandas as pd
import pytest

from app.errors import InvalidMethodInputError
from app.statistics.multiple_testing import adjust_p_values, run_tost_equivalence


def test_adjust_p_values_holm():
    raw = {"a": 0.01, "b": 0.04, "c": 0.2}
    result = adjust_p_values(raw, method="holm")
    assert result.p_values["a"] >= raw["a"]
    assert all(0 <= p <= 1 for p in result.p_values.values())


def test_adjust_invalid_method():
    with pytest.raises(InvalidMethodInputError):
        adjust_p_values({"a": 0.05}, method="invalid")


def test_tost_equivalence_two_groups():
    df = pd.DataFrame({
        "outcome": [10.0, 10.1, 9.9, 10.2, 20.0, 20.1, 19.9, 20.2],
        "group": ["A"] * 4 + ["B"] * 4,
    })
    close = run_tost_equivalence(df, "outcome", "group", margin=5.0)
    assert "overall" in close.p_values
