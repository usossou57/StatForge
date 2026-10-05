"""Tests pour les vérifications diagnostiques."""

from pathlib import Path

import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from app.diagnostics.checks import (
    check_influential_observations,
    check_multicollinearity,
    check_sample_size,
    check_separation,
)
from app.diagnostics.models import Severity


@pytest.fixture
def fixtures_dir() -> Path:
    """Retourne le chemin vers les fixtures de cas limites."""
    return Path(__file__).parent.parent / "fixtures" / "edge_cases"


def test_check_multicollinearity_severe(fixtures_dir: Path) -> None:
    """Teste la détection de multicolinéarité sévère (VIF > 10)."""
    df = pd.read_csv(fixtures_dir / "multicollinearity.csv")
    diagnostics = check_multicollinearity(df, predictors=["x1", "x2", "x3"])

    # Doit détecter au moins un CERTAIN_ISSUE
    assert len(diagnostics) > 0
    assert any(d.severity == Severity.CERTAIN_ISSUE for d in diagnostics)
    assert any(d.code == "HIGH_VIF" for d in diagnostics)


def test_check_multicollinearity_clean(fixtures_dir: Path) -> None:
    """Teste qu'aucun diagnostic n'est déclenché sur données propres."""
    df = pd.read_csv(fixtures_dir / "clean_data.csv")
    diagnostics = check_multicollinearity(df, predictors=["x1", "x2", "x3"])

    # Ne doit pas déclencher de CERTAIN_ISSUE
    assert not any(d.severity == Severity.CERTAIN_ISSUE for d in diagnostics)


def test_check_multicollinearity_single_predictor() -> None:
    """Teste qu'aucun diagnostic n'est déclenché avec un seul prédicteur."""
    df = pd.DataFrame({"y": [1, 2, 3], "x1": [1, 2, 3]})
    diagnostics = check_multicollinearity(df, predictors=["x1"])
    assert len(diagnostics) == 0


def test_check_separation_complete(fixtures_dir: Path) -> None:
    """Teste la détection de séparation complète."""
    df = pd.read_csv(fixtures_dir / "separation.csv")

    # Ajuster un modèle logistique sur des données séparées
    model = smf.logit("outcome ~ x1 + x2", data=df).fit(disp=0)
    diagnostics = check_separation(model)

    # Doit détecter soit NON_CONVERGENCE, soit QUASI_SEPARATION, soit COMPLETE_SEPARATION
    assert len(diagnostics) > 0
    assert any(
        d.code in ("NON_CONVERGENCE", "QUASI_SEPARATION", "COMPLETE_SEPARATION")
        for d in diagnostics
    )


def test_check_separation_clean() -> None:
    """Teste qu'aucun diagnostic n'est déclenché sur données propres."""
    # Dataset sans séparation
    df = pd.DataFrame({
        "outcome": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "x2": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    })
    model = smf.logit("outcome ~ x1 + x2", data=df).fit(disp=0)
    diagnostics = check_separation(model)

    # Ne doit pas déclencher de CERTAIN_ISSUE
    assert not any(d.severity == Severity.CERTAIN_ISSUE for d in diagnostics)


def test_check_influential_observations(fixtures_dir: Path) -> None:
    """Teste la détection d'observations influentes."""
    df = pd.read_csv(fixtures_dir / "influential.csv")

    # Ajuster un modèle OLS
    X = df[["x1", "x2"]]
    X = sm.add_constant(X)
    y = df["y"]
    model = sm.OLS(y, X).fit()

    diagnostics = check_influential_observations(model)

    # Doit détecter au moins un WARNING
    assert len(diagnostics) > 0
    assert any(d.code == "INFLUENTIAL_OBSERVATIONS" for d in diagnostics)
    assert any(d.severity == Severity.WARNING for d in diagnostics)


def test_check_influential_observations_clean() -> None:
    """Teste qu'aucun diagnostic n'est déclenché sur données propres."""
    df = pd.DataFrame({
        "y": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "x2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    })
    X = df[["x1", "x2"]]
    X = sm.add_constant(X)
    y = df["y"]
    model = sm.OLS(y, X).fit()

    diagnostics = check_influential_observations(model)

    # Ne doit pas déclencher de WARNING
    assert not any(d.severity == Severity.WARNING for d in diagnostics)


def test_check_sample_size_logistic_insufficient(fixtures_dir: Path) -> None:
    """Teste la détection de taille d'échantillon insuffisante pour logistique."""
    df = pd.read_csv(fixtures_dir / "small_sample.csv")
    diagnostics = check_sample_size(df, method="logistic_regression", n_predictors=2)

    # Doit détecter un WARNING (5 obs < 10 × 2 = 20 recommandé)
    assert len(diagnostics) > 0
    assert any(d.code == "LOW_SAMPLE_SIZE_LOGISTIC" for d in diagnostics)
    assert any(d.severity == Severity.WARNING for d in diagnostics)


def test_check_sample_size_logistic_sufficient() -> None:
    """Teste qu'aucun diagnostic n'est déclenché avec taille suffisante."""
    df = pd.DataFrame({
        "y": [0, 1] * 50,
        "x1": range(100),
        "x2": range(100, 200),
    })
    diagnostics = check_sample_size(df, method="logistic_regression", n_predictors=2)

    # 100 obs >= 10 × 2 = 20 recommandé
    assert len(diagnostics) == 0


def test_check_sample_size_linear_insufficient() -> None:
    """Teste la détection de taille d'échantillon insuffisante pour linéaire."""
    df = pd.DataFrame({
        "y": [1, 2, 3, 4, 5],
        "x1": [1, 2, 3, 4, 5],
        "x2": [2, 3, 4, 5, 6],
        "x3": [3, 4, 5, 6, 7],
    })
    diagnostics = check_sample_size(df, method="linear_regression", n_predictors=3)

    # 5 obs < 10 × 3 = 30 recommandé
    assert len(diagnostics) > 0
    assert any(d.code == "LOW_SAMPLE_SIZE_LINEAR" for d in diagnostics)


def test_check_sample_size_unknown_method() -> None:
    """Teste qu'aucun diagnostic n'est déclenché pour méthode inconnue."""
    df = pd.DataFrame({"y": [1, 2, 3], "x1": [1, 2, 3]})
    diagnostics = check_sample_size(df, method="unknown_method", n_predictors=1)
    assert len(diagnostics) == 0
