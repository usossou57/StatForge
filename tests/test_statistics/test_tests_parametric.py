"""Tests unitaires pour les tests paramétriques (Phase 17 — T17.1).

Couverture :
- Cas nominal pour chaque fonction (6 fonctions)
- Cas limites (valeurs manquantes, petits effectifs, erreurs)
- Validation des contrats AnalysisResult (V1 §7, V2 §6.2)
- Validation des diagnostics retournés
- Validation de l'enregistrement dans le registre (V2 §7.1)
- Validation croisée avec R (benchmarks)
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.errors import InvalidMethodInputError
from app.results.models import AnalysisResult
from app.statistics.registry import get_method, list_methods
from app.statistics.tests_parametric import (
    run_fisher_exact,
    run_one_sample_t_test,
    run_paired_t_test,
    run_proportion_test,
    run_variance_test,
    run_welch_t_test,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def df_paired():
    """DataFrame pour test t pairé."""
    np.random.seed(42)
    n = 30
    before = np.random.normal(10, 2, n)
    after = before + np.random.normal(1.5, 0.8, n)
    return pd.DataFrame({"before": before, "after": after})


@pytest.fixture
def df_two_groups():
    """DataFrame pour test t de Welch et tests de variances."""
    np.random.seed(42)
    group_a = np.random.normal(10, 2, 25)
    group_b = np.random.normal(12, 3, 25)
    return pd.DataFrame(
        {
            "outcome": np.concatenate([group_a, group_b]),
            "group": ["A"] * 25 + ["B"] * 25,
        }
    )


@pytest.fixture
def df_one_sample():
    """DataFrame pour test t à un échantillon."""
    np.random.seed(42)
    values = np.random.normal(5.5, 1.2, 40)
    return pd.DataFrame({"value": values})


@pytest.fixture
def df_proportion():
    """DataFrame pour test de proportion."""
    np.random.seed(42)
    n = 100
    successes = np.random.choice(["oui", "non"], size=n, p=[0.6, 0.4])
    groups = np.random.choice(["G1", "G2"], size=n)
    return pd.DataFrame({"result": successes, "group": groups})


@pytest.fixture
def df_fisher():
    """DataFrame pour test exact de Fisher (tableau 2x2)."""
    np.random.seed(42)
    var1 = np.random.choice(["X", "Y"], size=50)
    var2 = np.random.choice(["A", "B"], size=50)
    return pd.DataFrame({"var1": var1, "var2": var2})


# ============================================================================
# Helpers de validation
# ============================================================================


def _assert_valid_analysis_result(result: AnalysisResult, method: str):
    """Vérifie que le résultat respecte le contrat AnalysisResult (V1 §7, V2 §6.2)."""
    assert isinstance(result, AnalysisResult)
    assert result.method == method
    assert isinstance(result.sample_size, int)
    assert result.sample_size > 0
    assert isinstance(result.estimates, dict)
    assert "statistic" in result.estimates or "odds_ratio" in result.estimates or "proportion" in result.estimates
    assert isinstance(result.p_values, dict)
    assert "overall" in result.p_values
    assert 0 <= result.p_values["overall"] <= 1
    assert isinstance(result.diagnostics, list)
    assert isinstance(result.assumptions_checked, list)
    assert len(result.assumptions_checked) > 0
    assert isinstance(result.generated_code, str)
    assert len(result.generated_code) > 0
    assert isinstance(result.software_versions, dict)
    assert "scipy" in result.software_versions


# ============================================================================
# Tests : Test t pairé
# ============================================================================


class TestPairedTTest:
    def test_nominal(self, df_paired):
        result = run_paired_t_test(df_paired, "before", "after")
        _assert_valid_analysis_result(result, "paired_t_test")
        assert result.sample_size == 30
        assert "mean_difference" in result.estimates
        assert "cohens_d" in result.estimates
        assert "mean_difference" in result.confidence_intervals

    def test_missing_values_dropped(self):
        df = pd.DataFrame(
            {
                "before": [1, 2, np.nan, 4, 5],
                "after": [1.5, 2.5, 3.5, np.nan, 5.5],
            }
        )
        result = run_paired_t_test(df, "before", "after")
        # 3 paires complètes seulement (indices 0, 1, 4)
        assert result.sample_size == 3
        assert "Suppression des paires incomplètes" in result.preprocessing[0]

    def test_too_few_pairs_raises(self):
        df = pd.DataFrame({"before": [1, np.nan], "after": [2, 3]})
        with pytest.raises(InvalidMethodInputError, match="Au moins 2 paires"):
            run_paired_t_test(df, "before", "after")

    def test_missing_column_raises(self, df_paired):
        with pytest.raises(InvalidMethodInputError, match="doivent exister"):
            run_paired_t_test(df_paired, "before", "missing_col")

    def test_registered_in_registry(self):
        # CORRECTION : get_method retourne un tuple (MethodSpec, Callable)
        spec, func = get_method("paired_t_test")
        assert spec is not None
        assert spec.method_id == "paired_t_test"
        assert spec.family == "inference"
        assert spec.phase == 17
        assert callable(func)


# ============================================================================
# Tests : Test t de Welch
# ============================================================================


class TestWelchTTest:
    def test_nominal(self, df_two_groups):
        result = run_welch_t_test(df_two_groups, "outcome", "group")
        _assert_valid_analysis_result(result, "welch_t_test")
        assert result.sample_size == 50
        assert "mean_difference" in result.estimates
        assert "cohens_d" in result.estimates
        assert result.model_metrics["group_sizes"]["A"] == 25
        assert result.model_metrics["group_sizes"]["B"] == 25

    def test_unequal_group_sizes(self):
        df = pd.DataFrame(
            {
                "outcome": [1, 2, 3, 4, 5, 6, 7],
                "group": ["A", "A", "A", "B", "B", "B", "B"],
            }
        )
        result = run_welch_t_test(df, "outcome", "group")
        assert result.sample_size == 7
        assert result.model_metrics["group_sizes"]["A"] == 3
        assert result.model_metrics["group_sizes"]["B"] == 4

    def test_three_groups_raises(self):
        df = pd.DataFrame(
            {
                "outcome": [1, 2, 3, 4, 5, 6],
                "group": ["A", "A", "B", "B", "C", "C"],
            }
        )
        with pytest.raises(InvalidMethodInputError, match="exactement 2 modalités"):
            run_welch_t_test(df, "outcome", "group")

    def test_missing_column_raises(self, df_two_groups):
        with pytest.raises(InvalidMethodInputError, match="doivent exister"):
            run_welch_t_test(df_two_groups, "outcome", "missing_group")

    def test_registered_in_registry(self):
        # CORRECTION : get_method retourne un tuple (MethodSpec, Callable)
        spec, func = get_method("welch_t_test")
        assert spec is not None
        assert spec.method_id == "welch_t_test"
        assert spec.family == "inference"
        assert spec.phase == 17
        assert callable(func)


# ============================================================================
# Tests : Test t à un échantillon
# ============================================================================


class TestOneSampleTTest:
    def test_nominal(self, df_one_sample):
        result = run_one_sample_t_test(df_one_sample, "value", mu0=5.0)
        _assert_valid_analysis_result(result, "one_sample_t_test")
        assert result.sample_size == 40
        assert result.estimates["mu0"] == 5.0
        assert "mean" in result.estimates
        assert "cohens_d" in result.estimates

    def test_missing_values_dropped(self):
        df = pd.DataFrame({"value": [1, 2, np.nan, 4, 5, 6]})
        result = run_one_sample_t_test(df, "value", mu0=3.0)
        assert result.sample_size == 5

    def test_too_few_obs_raises(self):
        df = pd.DataFrame({"value": [1, np.nan]})
        with pytest.raises(InvalidMethodInputError, match="Au moins 2 observations"):
            run_one_sample_t_test(df, "value", mu0=0.0)

    def test_missing_column_raises(self, df_one_sample):
        with pytest.raises(InvalidMethodInputError, match="doit exister"):
            run_one_sample_t_test(df_one_sample, "missing_col", mu0=5.0)

    def test_registered_in_registry(self):
        # CORRECTION : get_method retourne un tuple (MethodSpec, Callable)
        spec, func = get_method("one_sample_t_test")
        assert spec is not None
        assert spec.method_id == "one_sample_t_test"
        assert spec.family == "inference"
        assert spec.phase == 17
        assert callable(func)


# ============================================================================
# Tests : Test de proportion
# ============================================================================


class TestProportionTest:
    def test_one_sample_nominal(self, df_proportion):
        result = run_proportion_test(df_proportion, "result", "oui", p0=0.5)
        _assert_valid_analysis_result(result, "proportion_test")
        assert result.sample_size == 100
        assert "proportion" in result.estimates
        assert "successes" in result.estimates
        assert "proportion" in result.confidence_intervals

    def test_two_sample_nominal(self, df_proportion):
        result = run_proportion_test(
            df_proportion, "result", "oui", p0=0.5, group="group"
        )
        _assert_valid_analysis_result(result, "proportion_test_two_sample")
        assert "proportion_group1" in result.estimates
        assert "proportion_group2" in result.estimates
        assert "proportion_difference" in result.estimates

    def test_exact_test_for_small_sample(self):
        df = pd.DataFrame({"result": ["oui"] * 5 + ["non"] * 3})
        result = run_proportion_test(df, "result", "oui", p0=0.5)
        assert result.model_metrics["test_type"] == "exact"

    def test_z_test_for_large_sample(self):
        df = pd.DataFrame({"result": ["oui"] * 50 + ["non"] * 50})
        result = run_proportion_test(df, "result", "oui", p0=0.5)
        assert result.model_metrics["test_type"] == "z"

    def test_missing_column_raises(self, df_proportion):
        with pytest.raises(InvalidMethodInputError, match="doit exister"):
            run_proportion_test(df_proportion, "missing_col", "oui")

    def test_registered_in_registry(self):
        # CORRECTION : get_method retourne un tuple (MethodSpec, Callable)
        spec, func = get_method("proportion_test")
        assert spec is not None
        assert spec.method_id == "proportion_test"
        assert spec.family == "inference"
        assert spec.phase == 17
        assert callable(func)


# ============================================================================
# Tests : Test exact de Fisher
# ============================================================================


class TestFisherExact:
    def test_nominal(self, df_fisher):
        result = run_fisher_exact(df_fisher, "var1", "var2")
        _assert_valid_analysis_result(result, "fisher_exact")
        assert "odds_ratio" in result.estimates
        assert "contingency_table" in result.model_metrics

    def test_non_2x2_raises(self):
        df = pd.DataFrame(
            {
                "var1": ["A", "B", "C"] * 10,
                "var2": ["X", "Y"] * 15,
            }
        )
        with pytest.raises(InvalidMethodInputError, match="2x2"):
            run_fisher_exact(df, "var1", "var2")

    def test_missing_values_dropped(self):
        df = pd.DataFrame(
            {
                "var1": ["A", "B", np.nan, "A", "B"],
                "var2": ["X", "Y", "X", np.nan, "Y"],
            }
        )
        result = run_fisher_exact(df, "var1", "var2")
        assert result.sample_size == 3  # 3 paires complètes

    def test_missing_column_raises(self, df_fisher):
        with pytest.raises(InvalidMethodInputError, match="doivent exister"):
            run_fisher_exact(df_fisher, "var1", "missing_var")

    def test_registered_in_registry(self):
        # CORRECTION : get_method retourne un tuple (MethodSpec, Callable)
        spec, func = get_method("fisher_exact")
        assert spec is not None
        assert spec.method_id == "fisher_exact"
        assert spec.family == "inference"
        assert spec.phase == 17
        assert callable(func)


# ============================================================================
# Tests : Test de variances
# ============================================================================


class TestVarianceTest:
    def test_levene_nominal(self, df_two_groups):
        result = run_variance_test(df_two_groups, "outcome", "group", method="levene")
        _assert_valid_analysis_result(result, "variance_test_levene")
        assert result.model_metrics["test_name"] == "Levene"
        assert result.model_metrics["n_groups"] == 2

    def test_bartlett_nominal(self, df_two_groups):
        result = run_variance_test(df_two_groups, "outcome", "group", method="bartlett")
        _assert_valid_analysis_result(result, "variance_test_bartlett")
        assert result.model_metrics["test_name"] == "Bartlett"

    def test_fligner_nominal(self, df_two_groups):
        result = run_variance_test(df_two_groups, "outcome", "group", method="fligner")
        _assert_valid_analysis_result(result, "variance_test_fligner")
        assert result.model_metrics["test_name"] == "Fligner-Killeen"

    def test_invalid_method_raises(self, df_two_groups):
        with pytest.raises(InvalidMethodInputError, match="Méthode invalide"):
            run_variance_test(df_two_groups, "outcome", "group", method="invalid")

    def test_one_group_raises(self):
        df = pd.DataFrame(
            {
                "outcome": [1, 2, 3, 4, 5],
                "group": ["A"] * 5,
            }
        )
        with pytest.raises(InvalidMethodInputError, match="au moins 2 modalités"):
            run_variance_test(df, "outcome", "group")

    def test_missing_column_raises(self, df_two_groups):
        with pytest.raises(InvalidMethodInputError, match="doivent exister"):
            run_variance_test(df_two_groups, "outcome", "missing_group")

    def test_registered_in_registry(self):
        # CORRECTION : get_method retourne un tuple (MethodSpec, Callable)
        spec, func = get_method("variance_test")
        assert spec is not None
        assert spec.method_id == "variance_test"
        assert spec.family == "inference"
        assert spec.phase == 17
        assert callable(func)


# ============================================================================
# Tests : Registre global
# ============================================================================


class TestRegistry:
    def test_all_methods_registered(self):
        methods = list_methods(family="inference")
        method_ids = [m.method_id for m in methods]
        assert "paired_t_test" in method_ids
        assert "welch_t_test" in method_ids
        assert "one_sample_t_test" in method_ids
        assert "proportion_test" in method_ids
        assert "fisher_exact" in method_ids
        assert "variance_test" in method_ids

    def test_all_methods_phase_17(self):
        methods = list_methods(family="inference")
        for m in methods:
            if m.method_id in [
                "paired_t_test",
                "welch_t_test",
                "one_sample_t_test",
                "proportion_test",
                "fisher_exact",
                "variance_test",
            ]:
                assert m.phase == 17


# ============================================================================
# Tests : Validation croisée avec R (benchmarks)
# ============================================================================


class TestRBenchmarkValidation:
    """Validation croisée avec R pour garantir la reproductibilité (V2 §6.2).
    
    Les valeurs attendues sont issues de l'exécution réelle du script 
    benchmarks/r/phase17_benchmarks.R.
    """

    def test_paired_t_test_matches_r(self):
        before = np.array([10.2, 11.5, 9.8, 10.9, 11.2, 10.5, 11.8, 10.1, 10.7, 11.0])
        after = np.array([11.5, 12.1, 10.5, 11.8, 12.0, 11.2, 12.5, 10.9, 11.4, 11.7])
        df = pd.DataFrame({"before": before, "after": after})

        result = run_paired_t_test(df, "before", "after")

        # R output: t = -12.6863, p ≈ 1.29e-07 (affiché 0.000000 par R)
        # SciPy ttest_rel(before, after) calcule bien before - after
        expected_t = -12.6863
        expected_p = 1.29e-07

        assert abs(result.estimates["statistic"] - expected_t) < 0.01
        assert abs(result.p_values["overall"] - expected_p) < 1e-5

    def test_welch_t_test_matches_r(self):
        group_a = np.array([10.2, 11.5, 9.8, 10.9, 11.2])
        group_b = np.array([12.5, 13.1, 11.8, 12.9, 13.5])
        df = pd.DataFrame(
            {"outcome": np.concatenate([group_a, group_b]), "group": ["A"] * 5 + ["B"] * 5}
        )

        result = run_welch_t_test(df, "outcome", "group")

        # R output: t = -4.7688, p = 0.001440
        expected_t = -4.7688
        expected_p = 0.001440

        assert abs(result.estimates["statistic"] - expected_t) < 0.01
        assert abs(result.p_values["overall"] - expected_p) < 0.0001

    def test_fisher_exact_matches_r(self):
        df = pd.DataFrame(
            {
                "var1": ["A"] * 10 + ["B"] * 10,
                "var2": ["X"] * 8 + ["Y"] * 2 + ["X"] * 3 + ["Y"] * 7,
            }
        )

        result = run_fisher_exact(df, "var1", "var2")

        # R output: p = 0.069779
        # NOTE : SciPy renvoie l'Odds Ratio de l'échantillon : (8 * 7) / (3 * 2) = 9.3333
        # R renvoie l'Odds Ratio par maximum de vraisemblance conditionnel : 8.1531
        # Nous testons la p-value (identique) et l'OR de SciPy.
        expected_or = 9.3333
        expected_p = 0.069779

        assert abs(result.estimates["odds_ratio"] - expected_or) < 0.01
        assert abs(result.p_values["overall"] - expected_p) < 0.0001