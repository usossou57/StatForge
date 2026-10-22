import numpy as np
import pandas as pd
import pytest

from app.statistics.regression import (
    compare_models,
    run_firth_logistic,
    run_linear_regression,
    run_logistic_regression,
    run_multinomial_logistic_regression,
)


def test_run_multinomial_logistic_regression() -> None:
    """Teste la régression logistique multinomiale sur un dataset synthétique."""
    # Dataset avec 3 catégories de résultat
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "outcome": np.random.choice(["A", "B", "C"], size=300),
            "x1": np.random.randn(300),
            "x2": np.random.randn(300),
        }
    )
    # Rendons x1 prédictif de "B" vs "A"
    df.loc[df["outcome"] == "B", "x1"] += 1.5

    result = run_multinomial_logistic_regression(
        df, outcome="outcome", predictors=["x1", "x2"]
    )

    assert result.method == "multinomial_logistic_regression"
    assert result.sample_size == 300
    assert "aic" in result.model_metrics
    assert "bic" in result.model_metrics
    # Vérifie que les clés de estimates contiennent bien les modalités (ex: ('B', 'x1'))
    assert any("x1" in str(k) for k in result.estimates.keys())
    assert "import statsmodels.api as sm" in result.generated_code


def test_compare_models_valid() -> None:
    """Teste la comparaison de deux modèles de même famille."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "y": np.random.choice([0, 1], size=100),
            "x1": np.random.randn(100),
            "x2": np.random.randn(100),
        }
    )

    # Modèle 1 : un seul prédicteur
    res1 = run_logistic_regression(df, outcome="y", predictors=["x1"])
    # Modèle 2 : deux prédicteurs
    res2 = run_logistic_regression(df, outcome="y", predictors=["x1", "x2"])

    comparison = compare_models([res1, res2])

    assert comparison["family"] == "logistic_regression"
    assert comparison["n_models"] == 2
    assert len(comparison["comparison_table"]) == 2
    assert any(m["is_best"] for m in comparison["comparison_table"])
    # Le modèle avec plus de variables a généralement un AIC différent, l'un des deux est marqué is_best


def test_compare_models_empty_list() -> None:
    """Teste que compare_models lève une erreur sur liste vide."""
    with pytest.raises(ValueError, match="vide"):
        compare_models([])


def test_compare_models_mixed_families() -> None:
    """Teste que compare_models lève une erreur si les familles diffèrent."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "y_bin": np.random.choice([0, 1], size=100),
            "y_cont": np.random.randn(100),
            "x1": np.random.randn(100),
        }
    )

    res_logit = run_logistic_regression(df, outcome="y_bin", predictors=["x1"])
    res_ols = run_linear_regression(df, outcome="y_cont", predictors=["x1"])

    with pytest.raises(ValueError, match="même famille"):
        compare_models([res_logit, res_ols])


def test_run_firth_logistic() -> None:
    """Teste la régression de Firth (doit basculer sur R si firthlogist absent)."""
    # Dataset avec séparation quasi-parfaite pour forcer l'utilité de Firth
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "outcome": [0] * 10 + [1] * 10,
            "x1": list(range(1, 11)) + list(range(6, 16)),  # Séparation partielle
            "x2": np.random.randn(20),
        }
    )

    # Appel de la fonction avec le bon nom
    result = run_firth_logistic(df, outcome="outcome", predictors=["x1", "x2"])

    assert result.method == "firth_logistic_regression"
    assert result.sample_size == 20
    assert "log_likelihood" in result.model_metrics

    # Vérifie que les coefficients sont bien présents (soit "x1", soit "(Intercept)" selon R)
    assert "x1" in result.estimates or "(Intercept)" in result.estimates

    # Vérifie que le warning de bascule vers R est bien présent (car firthlogist n'est pas installé)
    assert any("moteur R" in w or "firthlogist" in w for w in result.warnings)
