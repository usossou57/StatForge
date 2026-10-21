"""Tests T17.3 — ANCOVA, MANOVA, post-hoc, Welch et mesures répétées.

Validation croisée R : lire benchmarks/r/outputs/phase17_anova_expected.json
produit par benchmarks/r/phase17_anova_benchmarks.R (données fixes partagées).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.errors import InvalidMethodInputError
from app.results.models import AnalysisResult
from app.statistics.registry import get_method
from app.statistics.tests_anova import (
    run_ancova,
    run_manova,
    run_posthoc,
    run_repeated_measures_anova,
    run_welch_anova,
)

R_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "benchmarks" / "r" / "outputs" / "phase17_anova_expected.json"
)
requires_r = pytest.mark.skipif(
    not R_OUTPUT.exists(),
    reason="Benchmark R non généré : exécuter Rscript benchmarks/r/phase17_anova_benchmarks.R",
)


# ---------------------------------------------------------------------------
# Données fixes partagées avec R (aucun tirage aléatoire)
# ---------------------------------------------------------------------------

WELCH_A = [10.2, 11.5, 9.8, 10.9, 11.2]
WELCH_B = [12.5, 13.1, 11.8, 12.9, 13.5]
WELCH_C = [14.5, 15.1, 13.8, 14.9, 15.5]

ANCOVA_COV = [10, 12, 14, 16, 18, 11, 13, 15, 17, 19, 12, 14, 16, 18, 20]
ANCOVA_OUT = [
    12.1, 13.4, 12.8, 14.2, 13.1,
    14.5, 15.2, 14.8, 16.1, 15.4,
    16.2, 17.1, 16.8, 18.2, 17.5,
]

MANOVA_Y1 = WELCH_A + WELCH_B + WELCH_C
MANOVA_Y2 = [
    20.1, 19.4, 21.2, 20.8, 21.5,
    22.4, 23.1, 22.8, 24.2, 23.5,
    25.1, 24.8, 26.2, 25.5, 26.8,
]

RM_MATRIX = [
    [10.2, 12.1, 13.4],
    [11.5, 12.8, 14.2],
    [9.8, 11.2, 12.9],
    [10.9, 12.5, 13.8],
    [11.2, 13.1, 14.5],
    [10.5, 11.8, 13.1],
]


def _rm_long():
    rows = []
    for subject, values in enumerate(RM_MATRIX, start=1):
        for condition, value in zip(["c1", "c2", "c3"], values):
            rows.append({"outcome": value, "condition": condition, "subject": subject})
    return pd.DataFrame(rows)


def _expected():
    return json.loads(R_OUTPUT.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_ancova():
    return pd.DataFrame({
        "outcome": ANCOVA_OUT,
        "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
        "covariate": ANCOVA_COV,
    })


@pytest.fixture
def df_manova():
    return pd.DataFrame({
        "outcome1": MANOVA_Y1,
        "outcome2": MANOVA_Y2,
        "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
    })


@pytest.fixture
def df_posthoc():
    return pd.DataFrame({
        "outcome": WELCH_A + WELCH_B + WELCH_C,
        "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
    })


@pytest.fixture
def df_welch():
    return pd.DataFrame({
        "outcome": WELCH_A + WELCH_B + WELCH_C,
        "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
    })


@pytest.fixture
def df_rm():
    return _rm_long()


def _assert_contract(result: AnalysisResult, method: str):
    assert isinstance(result, AnalysisResult)
    assert result.method == method
    assert isinstance(result.sample_size, int) and result.sample_size > 0
    assert "f_statistic" in result.estimates or "pillai_trace" in result.estimates
    assert "overall" in result.p_values
    assert 0.0 <= result.p_values["overall"] <= 1.0
    assert isinstance(result.diagnostics, list)
    assert result.assumptions_checked
    assert isinstance(result.generated_code, str) and result.generated_code
    assert isinstance(result.software_versions, dict)
    assert result.model_metrics["method_family"] == "inference"
    # JSON-safe
    json.dumps(result.model_metrics, allow_nan=False)
    json.dumps(result.estimates, allow_nan=False)
    json.dumps(result.p_values, allow_nan=False)


# ---------------------------------------------------------------------------
# ANCOVA
# ---------------------------------------------------------------------------


class TestANCOVA:
    def test_nominal(self, df_ancova):
        result = run_ancova(df_ancova, "outcome", "group", ["covariate"])
        _assert_contract(result, "ancova")
        assert result.sample_size == 15
        assert "partial_eta_squared" in result.estimates
        assert result.model_metrics["n_covariates"] == 1
        assert result.model_metrics["anova_type"] == 2
        assert 0.0 <= result.estimates["partial_eta_squared"] <= 1.0

    def test_missing_values_dropped(self):
        df = pd.DataFrame({
            "outcome": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "group": ["A", "A", "A", "B", "B", "B"],
            "covariate": [10, 20, 30, 40, 50, 60],
        })
        result = run_ancova(df, "outcome", "group", ["covariate"])
        assert result.sample_size == 5
        assert result.model_metrics["n_obs_dropped"] == 1

    def test_one_group_raises(self):
        df = pd.DataFrame({
            "outcome": [1.0, 2.0, 3.0, 4.0, 5.0],
            "group": ["A"] * 5,
            "covariate": [10, 20, 30, 40, 50],
        })
        with pytest.raises(InvalidMethodInputError, match="au moins 2 modalités"):
            run_ancova(df, "outcome", "group", ["covariate"])

    def test_no_covariate_raises(self, df_ancova):
        with pytest.raises(InvalidMethodInputError, match="Au moins une covariable"):
            run_ancova(df_ancova, "outcome", "group", [])

    def test_missing_column_raises(self, df_ancova):
        with pytest.raises(InvalidMethodInputError, match="doit exister"):
            run_ancova(df_ancova, "outcome", "missing_group", ["covariate"])

    def test_registered(self):
        spec, func = get_method("ancova")
        assert spec.method_id == "ancova" and spec.phase == 17 and callable(func)


# ---------------------------------------------------------------------------
# MANOVA
# ---------------------------------------------------------------------------


class TestMANOVA:
    def test_nominal(self, df_manova):
        result = run_manova(df_manova, ["outcome1", "outcome2"], "group")
        _assert_contract(result, "manova")
        assert result.sample_size == 15
        assert "pillai_trace" in result.estimates
        assert result.model_metrics["n_outcomes"] == 2
        assert 0.0 <= result.estimates["pillai_trace"] <= 2.0

    def test_one_outcome_raises(self):
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "group": ["A", "A", "B", "B"]})
        with pytest.raises(InvalidMethodInputError, match="Au moins 2 variables dépendantes"):
            run_manova(df, ["y"], "group")

    def test_one_group_raises(self, df_manova):
        df = df_manova.copy()
        df["group"] = "A"
        with pytest.raises(InvalidMethodInputError, match="au moins 2 modalités"):
            run_manova(df, ["outcome1", "outcome2"], "group")

    def test_missing_column_raises(self, df_manova):
        with pytest.raises(InvalidMethodInputError, match="doit exister"):
            run_manova(df_manova, ["outcome1", "outcome2"], "missing_group")

    def test_registered(self):
        spec, func = get_method("manova")
        assert spec.method_id == "manova" and callable(func)


# ---------------------------------------------------------------------------
# Post-hoc
# ---------------------------------------------------------------------------


class TestPosthoc:
    @pytest.mark.parametrize("method", ["tukey", "games-howell", "dunn"])
    def test_nominal(self, df_posthoc, method):
        result = run_posthoc(df_posthoc, "outcome", "group", method=method)
        _assert_contract(result, f"posthoc_{method}")
        assert result.model_metrics["n_groups"] == 3
        assert result.model_metrics["n_comparisons"] == 3
        comparisons = result.model_metrics["tables"]["comparisons"]
        assert {c["group1"] for c in comparisons} <= {"A", "B", "C"}
        assert all(c["p_adjusted"] is None or 0.0 <= c["p_adjusted"] <= 1.0 for c in comparisons)

    def test_tukey_confidence_intervals(self, df_posthoc):
        result = run_posthoc(df_posthoc, "outcome", "group", method="tukey")
        assert result.confidence_intervals
        for bounds in result.confidence_intervals.values():
            assert bounds[0] <= bounds[1]

    def test_invalid_method_raises(self, df_posthoc):
        with pytest.raises(InvalidMethodInputError, match="Méthode invalide"):
            run_posthoc(df_posthoc, "outcome", "group", method="invalid")

    def test_one_group_raises(self):
        df = pd.DataFrame({"outcome": [1.0, 2.0, 3.0], "group": ["A"] * 3})
        with pytest.raises(InvalidMethodInputError, match="au moins 2 modalités"):
            run_posthoc(df, "outcome", "group")

    def test_unequal_variances_diagnostic(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "outcome": np.concatenate([rng.normal(10, 1, 20), rng.normal(12, 6, 20)]),
            "group": ["A"] * 20 + ["B"] * 20,
        })
        result = run_posthoc(df, "outcome", "group", method="tukey")
        assert any(d.code == "UNEQUAL_VARIANCES" for d in result.diagnostics)

    def test_missing_column_raises(self, df_posthoc):
        with pytest.raises(InvalidMethodInputError, match="doivent exister"):
            run_posthoc(df_posthoc, "outcome", "missing_group")

    def test_registered(self):
        spec, func = get_method("posthoc")
        assert spec.method_id == "posthoc" and callable(func)


# ---------------------------------------------------------------------------
# Welch ANOVA
# ---------------------------------------------------------------------------


class TestWelchANOVA:
    def test_nominal(self, df_welch):
        result = run_welch_anova(df_welch, "outcome", "group")
        _assert_contract(result, "welch_anova")
        assert result.sample_size == 15
        assert "omega_squared" in result.estimates
        assert result.model_metrics["n_groups"] == 3
        assert 0.0 <= result.estimates["omega_squared"] <= 1.0

    def test_missing_values_dropped(self):
        df = pd.DataFrame({
            "outcome": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "group": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
        })
        result = run_welch_anova(df, "outcome", "group")
        assert result.sample_size == 8

    def test_one_group_raises(self):
        df = pd.DataFrame({"outcome": [1.0, 2.0, 3.0], "group": ["A"] * 3})
        with pytest.raises(InvalidMethodInputError, match="au moins 2 modalités"):
            run_welch_anova(df, "outcome", "group")

    def test_missing_column_raises(self, df_welch):
        with pytest.raises(InvalidMethodInputError, match="doivent exister"):
            run_welch_anova(df_welch, "outcome", "missing_group")

    def test_registered(self):
        spec, func = get_method("welch_anova")
        assert spec.method_id == "welch_anova" and callable(func)


# ---------------------------------------------------------------------------
# Mesures répétées
# ---------------------------------------------------------------------------


class TestRepeatedMeasuresANOVA:
    def test_nominal(self, df_rm):
        result = run_repeated_measures_anova(df_rm, "outcome", "condition", "subject")
        _assert_contract(result, "repeated_measures_anova")
        assert result.sample_size == 18
        assert result.model_metrics["n_subjects"] == 6
        assert result.model_metrics["n_conditions"] == 3

    def test_listwise_deletion_of_subjects(self, df_rm):
        df = df_rm.copy()
        df.loc[df["subject"] == 6, "outcome"] = np.nan
        result = run_repeated_measures_anova(df, "outcome", "condition", "subject")
        # Le sujet 6 est exclu en totalité : 5 sujets complets × 3 conditions
        assert result.sample_size == 15
        assert result.model_metrics["n_subjects"] == 5
        assert result.model_metrics["n_subjects_excluded"] == 1

    def test_insufficient_complete_subjects_raises(self, df_rm):
        df = df_rm.copy()
        df.loc[df["subject"] != 1, "outcome"] = np.nan
        with pytest.raises(InvalidMethodInputError, match="Au moins 2 sujets complets"):
            run_repeated_measures_anova(df, "outcome", "condition", "subject")

    def test_duplicate_subject_condition_raises(self, df_rm):
        df = pd.concat([df_rm, df_rm.iloc[[0]]], ignore_index=True)
        with pytest.raises(InvalidMethodInputError, match="doublons détectés"):
            run_repeated_measures_anova(df, "outcome", "condition", "subject")

    def test_one_condition_raises(self, df_rm):
        df = df_rm.copy()
        df["condition"] = "c1"
        with pytest.raises(InvalidMethodInputError, match="au moins 2 modalités"):
            run_repeated_measures_anova(df, "outcome", "condition", "subject")

    def test_missing_column_raises(self, df_rm):
        with pytest.raises(InvalidMethodInputError, match="doit exister"):
            run_repeated_measures_anova(df_rm, "outcome", "condition", "missing_subject")

    def test_registered(self):
        spec, func = get_method("repeated_measures_anova")
        assert spec.method_id == "repeated_measures_anova" and callable(func)


# ---------------------------------------------------------------------------
# Registre
# ---------------------------------------------------------------------------


class TestRegistry:
    @pytest.mark.parametrize(
        "method_id", ["ancova", "manova", "posthoc", "welch_anova", "repeated_measures_anova"]
    )
    def test_registered_phase_17(self, method_id):
        spec, func = get_method(method_id)
        assert spec.family == "inference"
        assert spec.phase == 17
        assert callable(func)


# ---------------------------------------------------------------------------
# Validation croisée R — données IDENTIQUES dans les deux langages
# ---------------------------------------------------------------------------


@requires_r
class TestRBenchmarkValidation:
    def test_welch_anova_matches_r(self):
        expected = _expected()["welch_anova"]
        df = pd.DataFrame({
            "outcome": WELCH_A + WELCH_B + WELCH_C,
            "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
        })
        result = run_welch_anova(df, "outcome", "group")
        assert result.estimates["f_statistic"] == pytest.approx(expected["F"], rel=1e-3, abs=1e-3)
        assert result.p_values["overall"] == pytest.approx(expected["p"], rel=1e-3, abs=1e-12)

    def test_ancova_matches_r(self, df_ancova):
        expected = _expected()["ancova"]
        result = run_ancova(df_ancova, "outcome", "group", ["covariate"])
        assert result.p_values["overall"] == pytest.approx(expected["group_p"], rel=1e-3, abs=1e-6)
        record = next(
            r for r in result.model_metrics["tables"]["ancova"] if r["term"] == "C(group)"
        )
        assert record["F"] == pytest.approx(expected["group_F"], rel=1e-3, abs=1e-3)

    def test_manova_matches_r(self, df_manova):
        expected = _expected()["manova"]
        result = run_manova(df_manova, ["outcome1", "outcome2"], "group")
        assert result.estimates["pillai_trace"] == pytest.approx(expected["pillai"], rel=1e-3, abs=1e-4)
        assert result.estimates["f_statistic"] == pytest.approx(expected["F"], rel=1e-2, abs=1e-2)
        assert result.p_values["overall"] == pytest.approx(expected["p"], rel=1e-2, abs=1e-4)

    def test_tukey_matches_r(self, df_posthoc):
        expected = _expected()["tukey"]
        result = run_posthoc(df_posthoc, "outcome", "group", method="tukey")
        comparisons = {
            (c["group1"], c["group2"]): c
            for c in result.model_metrics["tables"]["comparisons"]
        }
        for item in expected:
            g1, g2 = item["group1"], item["group2"]
            # R sort souvent "B-A", Python sort ("A", "B"). On normalise.
            if (g1, g2) in comparisons:
                actual = comparisons[(g1, g2)]
                expected_diff = item["mean_diff"]
            elif (g2, g1) in comparisons:
                actual = comparisons[(g2, g1)]
                expected_diff = item["mean_diff"]
            else:
                raise AssertionError(f"Paire manquante : ({g1}, {g2}) ou ({g2}, {g1})")
            
            assert actual["mean_diff"] == pytest.approx(expected_diff, rel=1e-3, abs=1e-3)
            assert actual["p_adjusted"] == pytest.approx(item["p_adj"], rel=1e-3, abs=1e-5)

    def test_repeated_measures_matches_r(self):
        expected = _expected()["repeated_measures_anova"]
        result = run_repeated_measures_anova(_rm_long(), "outcome", "condition", "subject")
        assert result.estimates["f_statistic"] == pytest.approx(expected["F"], rel=1e-3, abs=1e-3)
        assert result.p_values["overall"] == pytest.approx(expected["p"], rel=1e-2, abs=1e-8)

        sphericity = result.model_metrics["sphericity"]
        if expected.get("epsilon") is not None and sphericity["epsilon"] is not None:
            assert sphericity["epsilon"] == pytest.approx(expected["epsilon"], rel=1e-2, abs=1e-2)
        if expected.get("p_gg") is not None and "greenhouse_geisser" in result.p_values:
            assert result.p_values["greenhouse_geisser"] == pytest.approx(
                expected["p_gg"], rel=1e-2, abs=1e-6
            )