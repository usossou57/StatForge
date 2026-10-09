"""Tests pour le module app/statistics/normality_outliers.py.

Validation des tests de normalité et de la détection d'outliers.
Les valeurs attendues sont calibrées pour être cohérentes avec R (shapiro.test, outliers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.results.models import AnalysisResult
from app.statistics.normality_outliers import detect_outliers, run_normality_tests

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def df_normal() -> pd.DataFrame:
    """DataFrame avec une variable normalement distribuée."""
    np.random.seed(42)
    return pd.DataFrame({"x": np.random.randn(100)})


@pytest.fixture
def df_skewed() -> pd.DataFrame:
    """DataFrame avec une variable fortement asymétrique (non normale)."""
    np.random.seed(42)
    # Distribution exponentielle = très asymétrique
    return pd.DataFrame({"x": np.random.exponential(scale=2.0, size=100)})


@pytest.fixture
def df_with_outliers() -> pd.DataFrame:
    """DataFrame avec des outliers évidents."""
    np.random.seed(42)
    data = np.random.randn(100)
    # Ajout de 3 outliers très éloignés
    data[0] = 15.0
    data[1] = -12.0
    data[2] = 20.0
    return pd.DataFrame({"x": data})


@pytest.fixture
def df_multivariate() -> pd.DataFrame:
    """DataFrame multivarié pour Mahalanobis."""
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    y = 0.5 * x + np.random.randn(n) * 0.5
    # Ajout d'un outlier multivarié
    x_outlier = np.array([10.0, -10.0])
    x = np.concatenate([x, x_outlier])
    y = np.concatenate([y, np.array([10.0, -10.0])])
    return pd.DataFrame({"x": x, "y": y})


# ==============================================================================
# Tests pour run_normality_tests
# ==============================================================================


def test_normality_on_normal_data(df_normal: pd.DataFrame) -> None:
    """Sur des données normales, les tests ne doivent pas rejeter H0."""
    result = run_normality_tests(df_normal, "x")
    assert isinstance(result, AnalysisResult)
    assert result.method == "run_normality_tests"
    assert result.sample_size == 100

    # is_normal doit être True
    assert result.model_metrics["is_normal"] is True

    # Pas de diagnostic NON_NORMAL_DISTRIBUTION
    assert not any(d.code == "NON_NORMAL_DISTRIBUTION" for d in result.diagnostics)

    # Toutes les p-values doivent être > 0.05
    for p in result.p_values.values():
        assert p > 0.05


def test_normality_on_skewed_data(df_skewed: pd.DataFrame) -> None:
    """Sur des données asymétriques, au moins un test doit rejeter H0."""
    result = run_normality_tests(df_skewed, "x")
    assert isinstance(result, AnalysisResult)

    # is_normal doit être False
    assert result.model_metrics["is_normal"] is False

    # Diagnostic NON_NORMAL_DISTRIBUTION doit être présent
    codes = [d.code for d in result.diagnostics]
    assert "NON_NORMAL_DISTRIBUTION" in codes

    # Au moins une p-value < 0.05
    assert any(p < 0.05 for p in result.p_values.values())


def test_normality_specific_methods(df_normal: pd.DataFrame) -> None:
    """Teste l'exécution de méthodes spécifiques."""
    result = run_normality_tests(df_normal, "x", methods=["shapiro"])
    assert "shapiro" in result.model_metrics["tables"]["normality_tests"]
    assert "dagostino" not in result.model_metrics["tables"]["normality_tests"]
    assert "anderson" not in result.model_metrics["tables"]["normality_tests"]


def test_normality_invalid_method(df_normal: pd.DataFrame) -> None:
    """Teste qu'une méthode invalide lève ValueError."""
    with pytest.raises(ValueError, match="Méthodes invalides"):
        run_normality_tests(df_normal, "x", methods=["invalid"])


def test_normality_missing_column(df_normal: pd.DataFrame) -> None:
    """Teste qu'une colonne manquante lève ValueError."""
    with pytest.raises(ValueError, match="n'existe pas"):
        run_normality_tests(df_normal, "inexistante")


def test_normality_non_numeric_column() -> None:
    """Teste qu'une colonne non numérique lève ValueError."""
    df = pd.DataFrame({"cat": ["A", "B", "C", "A", "B"]})
    with pytest.raises(ValueError, match="doit être numérique"):
        run_normality_tests(df, "cat")


def test_normality_small_sample_warning() -> None:
    """Teste qu'un warning est émis pour n < 3 sur Shapiro."""
    df = pd.DataFrame({"x": [1.0, 2.0]})
    result = run_normality_tests(df, "x", methods=["shapiro"])
    assert any("n < 3" in w for w in result.warnings)


def test_normality_large_sample_warning() -> None:
    """Teste qu'un warning est émis pour n > 5000 sur Shapiro."""
    np.random.seed(42)
    df = pd.DataFrame({"x": np.random.randn(6000)})
    result = run_normality_tests(df, "x", methods=["shapiro"])
    assert any("n > 5000" in w for w in result.warnings)


def test_normality_descriptive_stats(df_normal: pd.DataFrame) -> None:
    """Teste que les statistiques descriptives sont présentes."""
    result = run_normality_tests(df_normal, "x")
    desc = result.estimates["descriptive"]
    assert "n" in desc
    assert "mean" in desc
    assert "std" in desc
    assert "skewness" in desc
    assert "kurtosis" in desc
    assert desc["n"] == 100


def test_normality_anderson_critical_value(df_normal: pd.DataFrame) -> None:
    """Teste que le résultat d'Anderson contient bien les clés attendues."""
    result = run_normality_tests(df_normal, "x", methods=["anderson"])
    anderson_res = result.model_metrics["tables"]["normality_tests"]["anderson"]
    assert "statistic" in anderson_res
    assert "p_value" in anderson_res
    assert "critical_value_5pct" in anderson_res


# ==============================================================================
# Tests pour detect_outliers - IQR
# ==============================================================================


def test_outliers_iqr_basic(df_with_outliers: pd.DataFrame) -> None:
    """Teste la détection d'outliers par IQR."""
    result = detect_outliers(df_with_outliers, ["x"], method="iqr")
    assert isinstance(result, AnalysisResult)
    assert result.method == "detect_outliers"

    # Au moins 3 outliers doivent être détectés (indices 0, 1, 2)
    assert result.estimates["n_outliers"] >= 3

    # Diagnostic OUTLIERS_DETECTED doit être présent
    codes = [d.code for d in result.diagnostics]
    assert "OUTLIERS_DETECTED" in codes


def test_outliers_iqr_no_outliers(df_normal: pd.DataFrame) -> None:
    """Sur des données normales, peu ou pas d'outliers IQR."""
    result = detect_outliers(df_normal, ["x"], method="iqr")
    # Sur 100 observations normales, on s'attend à peu d'outliers IQR (< 5)
    assert result.estimates["n_outliers"] < 5


def test_outliers_iqr_details(df_with_outliers: pd.DataFrame) -> None:
    """Teste que les détails (Q1, Q3, IQR, bornes) sont présents."""
    result = detect_outliers(df_with_outliers, ["x"], method="iqr")
    details = result.model_metrics["tables"]["outlier_details"]["x"]
    assert "q1" in details
    assert "q3" in details
    assert "iqr" in details
    assert "lower_bound" in details
    assert "upper_bound" in details
    assert details["q1"] < details["q3"]


def test_outliers_iqr_custom_threshold(df_with_outliers: pd.DataFrame) -> None:
    """Teste qu'un seuil personnalisé modifie le nombre d'outliers."""
    result_strict = detect_outliers(
        df_with_outliers, ["x"], method="iqr", threshold=1.0
    )
    result_loose = detect_outliers(df_with_outliers, ["x"], method="iqr", threshold=3.0)
    # Un seuil plus strict (1.0) doit détecter plus d'outliers qu'un seuil large (3.0)
    assert result_strict.estimates["n_outliers"] >= result_loose.estimates["n_outliers"]


# ==============================================================================
# Tests pour detect_outliers - Z-score
# ==============================================================================


def test_outliers_zscore_basic(df_with_outliers: pd.DataFrame) -> None:
    """Teste la détection d'outliers par z-score."""
    result = detect_outliers(df_with_outliers, ["x"], method="zscore")
    assert isinstance(result, AnalysisResult)
    # Les 3 outliers extrêmes doivent être détectés
    assert result.estimates["n_outliers"] >= 3


def test_outliers_zscore_no_outliers(df_normal: pd.DataFrame) -> None:
    """Sur des données normales, peu d'outliers z-score."""
    result = detect_outliers(df_normal, ["x"], method="zscore")
    # Sur 100 observations normales, on s'attend à ~0 outliers z-score (seuil 3)
    assert result.estimates["n_outliers"] <= 2


def test_outliers_zscore_constant_column() -> None:
    """Teste qu'une colonne constante ne produit pas d'outliers."""
    df = pd.DataFrame({"x": [5.0] * 20})
    result = detect_outliers(df, ["x"], method="zscore")
    assert result.estimates["n_outliers"] == 0


# ==============================================================================
# Tests pour detect_outliers - Mahalanobis
# ==============================================================================


def test_outliers_mahalanobis_basic(df_multivariate: pd.DataFrame) -> None:
    """Teste la détection d'outliers multivariés par Mahalanobis."""
    result = detect_outliers(df_multivariate, ["x", "y"], method="mahalanobis")
    assert isinstance(result, AnalysisResult)
    # L'observation outlier ajoutée doit être détectée
    assert result.estimates["n_outliers"] >= 1


def test_outliers_mahalanobis_requires_two_variables(df_normal: pd.DataFrame) -> None:
    """Teste que Mahalanobis lève ValueError si une seule variable."""
    with pytest.raises(ValueError, match="au moins 2 variables"):
        detect_outliers(df_normal, ["x"], method="mahalanobis")


def test_outliers_mahalanobis_details(df_multivariate: pd.DataFrame) -> None:
    """Teste que les détails Mahalanobis sont présents."""
    result = detect_outliers(df_multivariate, ["x", "y"], method="mahalanobis")
    details = result.model_metrics["tables"]["outlier_details"]
    assert details["method"] == "mahalanobis"
    assert "threshold" in details
    assert "max_distance_squared" in details


# ==============================================================================
# Tests communs
# ==============================================================================


def test_outliers_invalid_method(df_normal: pd.DataFrame) -> None:
    """Teste qu'une méthode invalide lève ValueError."""
    with pytest.raises(ValueError, match="non reconnue"):
        detect_outliers(df_normal, ["x"], method="invalid")


def test_outliers_missing_column(df_normal: pd.DataFrame) -> None:
    """Teste qu'une colonne manquante lève ValueError."""
    with pytest.raises(ValueError, match="manquantes"):
        detect_outliers(df_normal, ["inexistante"])


def test_outliers_non_numeric_column() -> None:
    """Teste qu'une colonne non numérique lève ValueError."""
    df = pd.DataFrame({"cat": ["A", "B", "C"]})
    with pytest.raises(ValueError, match="non numériques"):
        detect_outliers(df, ["cat"])


def test_outliers_multiple_columns(df_with_outliers: pd.DataFrame) -> None:
    """Teste la détection sur plusieurs colonnes."""
    df_with_outliers["y"] = df_with_outliers["x"] * 2
    result = detect_outliers(df_with_outliers, ["x", "y"], method="iqr")
    # Les deux colonnes doivent avoir des outliers détectés
    details = result.model_metrics["tables"]["outlier_details"]
    assert "x" in details
    assert "y" in details
    assert details["x"]["n_outliers"] >= 3
    assert details["y"]["n_outliers"] >= 3
