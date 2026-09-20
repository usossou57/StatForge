"""Tests pour app/statistics/regression.py."""

import math

import pandas as pd
import pytest

from app.statistics.regression import run_logistic, run_ols


def test_ols_cas_nominal() -> None:
    """Teste un cas nominal de régression OLS."""
    df = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x2": [2.0, 1.0, 3.0, 2.0, 4.0]
    })
    result = run_ols(df, "y", ["x1", "x2"])
    assert result.method == "ols"
    assert "r_squared" in result.model_metrics
    assert "intercept" in result.estimates


def test_ols_colonne_manquante() -> None:
    """Teste qu'une ValueError est levée si une colonne manque."""
    df = pd.DataFrame({"y": [1.0, 2.0], "x1": [1.0, 2.0]})
    with pytest.raises(ValueError, match="n'existe pas"):
        run_ols(df, "y", ["x1", "manquante"])


def test_ols_valeurs_manquantes_warning() -> None:
    """Teste qu'un warning est ajouté quand des valeurs manquantes sont supprimées."""
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, None], "x": [1.0, 2.0, 3.0, 4.0]})
    result = run_ols(df, "y", ["x"])
    assert len(result.warnings) == 1
    assert "dropna" in result.preprocessing


def test_ols_validation_r() -> None:
    """Validation numérique OLS contre R (lm).
    
    Données R :
    > y <- c(1, 2, 3, 4, 5)
    > x <- c(1, 2, 3, 4, 5)
    > df <- data.frame(y, x)
    > model <- lm(y ~ x, data=df)
    > coef(model)
    (Intercept)           x 
              0           1 
    > summary(model)$r.squared
    [1] 1
    """
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = run_ols(df, "y", ["x"])
    
    # Utilisation de abs_tol car la valeur attendue est 0.0
    assert math.isclose(result.estimates["intercept"], 0.0, abs_tol=1e-6)
    assert math.isclose(result.estimates["x"], 1.0, rel_tol=1e-6)
    assert math.isclose(result.model_metrics["r_squared"], 1.0, rel_tol=1e-6)


def test_logistic_cas_nominal() -> None:
    """Teste un cas nominal de régression logistique."""
    df = pd.DataFrame({
        "y": [0, 0, 1, 0, 1, 1],
        "x": [1.0, 2.0, 2.5, 3.0, 4.0, 5.0]
    })
    result = run_logistic(df, "y", ["x"])
    assert result.method == "logistic"
    assert "pseudo_r_squared" in result.model_metrics
    assert "odds_ratios" in result.model_metrics


def test_logistic_non_binaire_valueerror() -> None:
    """Teste qu'une ValueError est levée si outcome n'est pas binaire."""
    df = pd.DataFrame({"y": [0, 1, 2], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="doit être binaire"):
        run_logistic(df, "y", ["x"])


def test_logistic_valeurs_manquantes_warning() -> None:
    """Teste qu'un warning est ajouté quand des valeurs manquantes sont supprimées."""
    # Dataset sans séparation parfaite après dropna
    df = pd.DataFrame({"y": [0, 1, 0, None], "x": [1.0, 2.0, 1.5, 3.0]})
    result = run_logistic(df, "y", ["x"])
    assert any("valeurs manquantes" in w.lower() for w in result.warnings)
    assert "dropna" in result.preprocessing


def test_logistic_validation_r() -> None:
    """Validation numérique logistique contre R (glm).
    
    Données R (sans séparation parfaite) :
    > y <- c(0, 0, 0, 1, 1, 1)
    > x <- c(1, 2, 3, 2, 3, 4)
    > df <- data.frame(y, x)
    > model <- glm(y ~ x, family=binomial, data=df)
    > coef(model)
    (Intercept)           x 
     -3.465736    1.386294 
    > exp(coef(model)["x"])
           x 
    4.000000 
    """
    df = pd.DataFrame({
        "y": [0, 0, 0, 1, 1, 1],
        "x": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0]
    })
    result = run_logistic(df, "y", ["x"])
    
    assert math.isclose(result.estimates["intercept"], -3.465736, rel_tol=1e-4)
    assert math.isclose(result.estimates["x"], 1.386294, rel_tol=1e-4)
    assert math.isclose(result.model_metrics["odds_ratios"]["x"], 4.0, rel_tol=1e-3)
