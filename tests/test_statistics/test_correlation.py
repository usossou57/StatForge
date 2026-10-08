"""Tests pour le module app/statistics/correlation.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.results.models import AnalysisResult
from app.statistics.correlation import (
    run_correlation,
    run_covariance,
    run_partial_correlation,
)


@pytest.fixture
def df_corr() -> pd.DataFrame:
    """DataFrame avec des variables corrélées pour les tests."""
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    y = 0.8 * x + np.random.randn(n) * 0.5
    z = np.random.randn(n)
    w = 0.3 * x + 0.5 * z + np.random.randn(n) * 0.3
    return pd.DataFrame({"x": x, "y": y, "z": z, "w": w})


# ==============================================================================
# Tests pour run_correlation
# ==============================================================================


def test_correlation_pearson_basic(df_corr: pd.DataFrame) -> None:
    """Teste la corrélation de Pearson entre deux variables."""
    result = run_correlation(df_corr, ["x", "y"], method="pearson")
    assert isinstance(result, AnalysisResult)
    assert result.method == "run_correlation"
    assert result.sample_size == 100

    corr = result.model_metrics["tables"]["correlation"]
    r_xy = corr["x"]["y"]
    # On s'attend à une corrélation forte positive (~0.8)
    assert r_xy > 0.6

    # p-value doit être très petite
    pval = result.model_metrics["tables"]["pvalue"]["x"]["y"]
    assert pval < 0.001

    # IC doit exister et encadrer r
    ci_lower = result.model_metrics["tables"]["ci_lower"]["x"]["y"]
    ci_upper = result.model_metrics["tables"]["ci_upper"]["x"]["y"]
    assert ci_lower < r_xy < ci_upper


def test_correlation_spearman(df_corr: pd.DataFrame) -> None:
    """Teste la corrélation de Spearman."""
    result = run_correlation(df_corr, ["x", "y"], method="spearman")
    assert isinstance(result, AnalysisResult)
    corr = result.model_metrics["tables"]["correlation"]
    assert corr["x"]["y"] > 0.6


def test_correlation_kendall(df_corr: pd.DataFrame) -> None:
    """Teste la corrélation de Kendall."""
    result = run_correlation(df_corr, ["x", "y"], method="kendall")
    assert isinstance(result, AnalysisResult)
    corr = result.model_metrics["tables"]["correlation"]
    assert corr["x"]["y"] > 0.4


def test_correlation_matrix_symmetry(df_corr: pd.DataFrame) -> None:
    """Teste que la matrice de corrélation est symétrique."""
    result = run_correlation(df_corr, ["x", "y", "z"])
    corr = result.model_metrics["tables"]["correlation"]
    assert corr["x"]["y"] == corr["y"]["x"]
    assert corr["x"]["z"] == corr["z"]["x"]


def test_correlation_diagonal_is_one(df_corr: pd.DataFrame) -> None:
    """Teste que la diagonale de la matrice vaut 1."""
    result = run_correlation(df_corr, ["x", "y", "z"])
    corr = result.model_metrics["tables"]["correlation"]
    for v in ["x", "y", "z"]:
        assert corr[v][v] == pytest.approx(1.0)


def test_correlation_invalid_method(df_corr: pd.DataFrame) -> None:
    """Teste qu'une méthode invalide lève ValueError."""
    with pytest.raises(ValueError, match="non reconnue"):
        run_correlation(df_corr, ["x", "y"], method="invalid")


def test_correlation_missing_column(df_corr: pd.DataFrame) -> None:
    """Teste qu'une colonne manquante lève ValueError."""
    with pytest.raises(ValueError, match="manquantes"):
        run_correlation(df_corr, ["x", "inexistante"])


# ==============================================================================
# Tests pour run_covariance
# ==============================================================================


def test_covariance_basic(df_corr: pd.DataFrame) -> None:
    """Teste le calcul de la matrice de covariance."""
    result = run_covariance(df_corr, ["x", "y"])
    assert isinstance(result, AnalysisResult)
    assert result.method == "run_covariance"

    cov = result.model_metrics["tables"]["covariance"]
    # La variance de x doit être positive
    assert cov["x"]["x"] > 0
    # La covariance x-y doit être positive (car corrélées positivement)
    assert cov["x"]["y"] > 0


def test_covariance_symmetry(df_corr: pd.DataFrame) -> None:
    """Teste que la matrice de covariance est symétrique."""
    result = run_covariance(df_corr, ["x", "y", "z"])
    cov = result.model_metrics["tables"]["covariance"]
    assert cov["x"]["y"] == pytest.approx(cov["y"]["x"])


def test_covariance_missing_column(df_corr: pd.DataFrame) -> None:
    """Teste qu'une colonne manquante lève ValueError."""
    with pytest.raises(ValueError, match="manquantes"):
        run_covariance(df_corr, ["x", "inexistante"])


# ==============================================================================
# Tests pour run_partial_correlation
# ==============================================================================


def test_partial_correlation_basic(df_corr: pd.DataFrame) -> None:
    """Teste la corrélation partielle entre x et w en contrôlant z."""
    result = run_partial_correlation(df_corr, x="x", y="w", controls=["z"])
    assert isinstance(result, AnalysisResult)
    assert result.method == "run_partial_correlation"

    table = result.model_metrics["tables"]["partial_correlation"]
    assert table["x"] == "x"
    assert table["y"] == "w"
    # x et w sont corrélés via x, même en contrôlant z
    assert abs(table["r"]) > 0.1


def test_partial_correlation_no_controls(df_corr: pd.DataFrame) -> None:
    """Teste que sans contrôle, on retrouve la corrélation simple."""
    result = run_partial_correlation(df_corr, x="x", y="y", controls=[])
    r_partial = result.estimates["r_partial"]

    # Comparer avec la corrélation simple de Pearson
    r_simple = df_corr["x"].corr(df_corr["y"])
    assert r_partial == pytest.approx(r_simple, abs=1e-10)


def test_partial_correlation_controls_for_confounding(df_corr: pd.DataFrame) -> None:
    """Teste l'effet du contrôle d'une variable sur la corrélation.

    Dans ce fixture, w = 0.3*x + 0.5*z + noise.
    Contrôler z retire le bruit indépendant de x, donc la corrélation partielle
    x-w|z devrait être plus forte que la corrélation simple x-w.
    """
    r_simple = df_corr["x"].corr(df_corr["w"])
    result = run_partial_correlation(df_corr, x="x", y="w", controls=["z"])
    r_partial = result.estimates["r_partial"]
    # En contrôlant z (qui ajoute du bruit à w), la corrélation augmente
    assert abs(r_partial) > abs(r_simple)


def test_partial_correlation_missing_column(df_corr: pd.DataFrame) -> None:
    """Teste qu'une colonne manquante lève ValueError."""
    with pytest.raises(ValueError, match="manquantes"):
        run_partial_correlation(df_corr, x="x", y="y", controls=["inexistante"])


def test_partial_correlation_ci_exists(df_corr: pd.DataFrame) -> None:
    """Teste que l'IC est calculé et cohérent."""
    result = run_partial_correlation(df_corr, x="x", y="y", controls=["z"])
    ci = result.confidence_intervals["r_partial"]
    r = result.estimates["r_partial"]
    assert ci[0] < r < ci[1]
