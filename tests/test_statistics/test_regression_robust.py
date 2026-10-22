"""Tests pour la régression robuste, polynomiale et par splines.
Inclut validation contre R (V2 §6.3, T12).
"""
import numpy as np
import pandas as pd
import pytest

from app.statistics.regression_robust import (
    run_polynomial_regression,
    run_robust_regression,
    run_spline_regression,
)


@pytest.fixture
def non_linear_data():
    """Données avec relation non linéaire et outliers pour tests."""
    np.random.seed(42)
    n = 150
    x = np.linspace(-3, 3, n)
    
    # Relation quadratique + bruit
    y = 0.5 * x**2 - 1.0 * x + 2.0 + np.random.randn(n) * 0.5
    
    # Ajouter quelques outliers
    y[10] += 15.0
    y[140] -= 15.0
    
    df = pd.DataFrame({"y": y, "x": x})
    df.to_csv("validation/data_robust_python.csv", index=False)
    return df


class TestRobustRegression:
    def test_fit_robust(self, non_linear_data):
        result = run_robust_regression(non_linear_data, "y", ["x"], m_estimator="huber")
        assert result.method == "robust_regression"
        assert "Intercept" in result.estimates
        assert "x" in result.estimates
        assert result.model_metrics["m_estimator"] == "huber"

    def test_robust_vs_r(self, non_linear_data):
        """Validation contre R : MASS::rlm(y ~ x, psi=psi.huber)
        Tolérance rel=1e-4 (optimisation itérative, V2 §6.3).
        """
        result = run_robust_regression(non_linear_data, "y", ["x"], m_estimator="huber")
        
        # Valeurs exactes extraites de R
        expected_intercept = 3.361839
        expected_x = -1.006502
        
        assert abs(result.estimates["Intercept"] - expected_intercept) < 1e-4
        assert abs(result.estimates["x"] - expected_x) < 1e-4
        assert result.model_metrics["n_obs"] == 150


class TestPolynomialRegression:
    def test_fit_polynomial(self, non_linear_data):
        result = run_polynomial_regression(non_linear_data, "y", "x", degree=2)
        assert result.method == "polynomial_regression"
        # statsmodels ajoute des espaces autour de '**' dans les noms de clés
        assert "I(x ** 2)" in result.estimates
        assert result.model_metrics["degree"] == 2
        assert result.model_metrics["r2"] > 0.5

    def test_polynomial_vs_r(self, non_linear_data):
        """Validation contre R : lm(y ~ poly(x, 2, raw=TRUE))
        Tolérance rel=1e-6 (formule fermée, V2 §6.3).
        """
        result = run_polynomial_regression(non_linear_data, "y", "x", degree=2)
        
        # Valeurs exactes extraites de R
        expected_intercept = 1.9732885
        expected_x = -1.1610739
        expected_x2 = 0.4952465
        
        assert abs(result.estimates["Intercept"] - expected_intercept) < 1e-6
        assert abs(result.estimates["x"] - expected_x) < 1e-6
        assert abs(result.estimates["I(x ** 2)"] - expected_x2) < 1e-6


class TestSplineRegression:
    def test_fit_spline(self, non_linear_data):
        result = run_spline_regression(non_linear_data, "y", "x", df_spline=4, degree=3)
        assert result.method == "spline_regression"
        assert result.model_metrics["df_spline"] == 4
        assert result.model_metrics["degree"] == 3
        # Vérifier que des coefficients de splines sont présents
        assert any("bs(x" in key for key in result.estimates.keys())
        assert result.model_metrics["r2"] > 0.5

    def test_spline_vs_r(self, non_linear_data):
        """Validation contre R : lm(y ~ splines::bs(x, df=4, degree=3))
        Tolérance rel=1e-5 (formule fermée, V2 §6.3).
        """
        result = run_spline_regression(non_linear_data, "y", "x", df_spline=4, degree=3)
        
        # Valeurs exactes extraites de R
        expected_intercept = 10.564559
        expected_bs0 = -5.540489
        expected_bs1 = -9.825911
        expected_bs2 = -9.003529
        expected_bs3 = -8.108758
        
        # sm.OLS avec sm.add_constant() utilise 'const' pour l'intercept
        assert abs(result.estimates["const"] - expected_intercept) < 1e-5
        
        # Les clés patsy sont typiquement "bs(x, df=4, degree=3)[0]", etc.
        # On récupère les clés contenant "bs(x" et on les trie pour garantir l'ordre
        bs_keys = sorted([k for k in result.estimates.keys() if "bs(x" in k])
        assert len(bs_keys) == 4, f"Attendu 4 clés de spline, trouvé: {bs_keys}"
        
        assert abs(result.estimates[bs_keys[0]] - expected_bs0) < 1e-5
        assert abs(result.estimates[bs_keys[1]] - expected_bs1) < 1e-5
        assert abs(result.estimates[bs_keys[2]] - expected_bs2) < 1e-5
        assert abs(result.estimates[bs_keys[3]] - expected_bs3) < 1e-5
        
        assert abs(result.model_metrics["r2"] - 0.6637628) < 1e-5