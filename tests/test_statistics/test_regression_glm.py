"""Tests pour les GLM (probit, Poisson, binomiale négative, Gamma).
Inclut validation contre R (V2 §6.3, T12).
"""
import numpy as np
import pandas as pd
import pytest

from app.statistics.regression_glm import (
    run_gamma_regression,
    run_negative_binomial_regression,
    run_poisson_regression,
    run_probit_regression,
)


# ============================================================================
# Données de test (graine fixe pour reproductibilité)
# ============================================================================


@pytest.fixture
def binary_data():
    """Données binaires pour probit."""
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    prob = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
    y = np.random.binomial(1, prob)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    # Sauvegarder pour validation R
    df.to_csv("validation/data_binary_python.csv", index=False)
    return df


@pytest.fixture
def count_data():
    """Données de comptage pour Poisson et binomiale négative."""
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    lam = np.exp(0.5 + 0.3 * x1 + 0.2 * x2)
    y = np.random.poisson(lam)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    # Sauvegarder pour validation R
    df.to_csv("validation/data_count_python.csv", index=False)
    return df


@pytest.fixture
def positive_data():
    """Données continues positives pour Gamma."""
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    scale = np.exp(1 + 0.3 * x1 + 0.2 * x2)
    y = np.random.gamma(shape=2, scale=scale / 2)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    # Sauvegarder pour validation R
    df.to_csv("validation/data_positive_python.csv", index=False)
    return df


# ============================================================================
# Tests Probit
# ============================================================================


class TestProbitRegression:
    def test_fit_probit(self, binary_data):
        result = run_probit_regression(binary_data, "y", ["x1", "x2"], link="probit")
        assert result.method == "probit_regression"
        assert "Intercept" in result.estimates
        assert "x1" in result.estimates
        assert "x2" in result.estimates
        assert result.sample_size == 200
        assert "aic" in result.model_metrics

    def test_probit_vs_r(self, binary_data):
        """Validation contre R : glm(y ~ x1 + x2, family=binomial(link="probit"))"""
        result = run_probit_regression(binary_data, "y", ["x1", "x2"], link="probit")

        # Valeurs exactes extraites de R (r_validation_python_data.R)
        expected_intercept = 0.03825044
        expected_x1 = 0.31263762
        expected_x2 = 0.29946622

        # Tolérance pour validation croisée (rel=1e-4 pour GLM itératif, V2 §6.3)
        assert abs(result.estimates["Intercept"] - expected_intercept) < 1e-4
        assert abs(result.estimates["x1"] - expected_x1) < 1e-4
        assert abs(result.estimates["x2"] - expected_x2) < 1e-4


# ============================================================================
# Tests Poisson
# ============================================================================


class TestPoissonRegression:
    def test_fit_poisson(self, count_data):
        result = run_poisson_regression(count_data, "y", ["x1", "x2"])
        assert result.method == "poisson_regression"
        assert "Intercept" in result.estimates
        assert "log_likelihood" in result.model_metrics
        assert result.sample_size == 200
        assert "incidence_rate_ratios" in result.model_metrics

    def test_poisson_vs_r(self, count_data):
        """Validation contre R : glm(y ~ x1 + x2, family=poisson)"""
        result = run_poisson_regression(count_data, "y", ["x1", "x2"])

        # Valeurs exactes extraites de R
        expected_intercept = 0.3907376
        expected_x1 = 0.3249596
        expected_x2 = 0.1394652

        assert abs(result.estimates["Intercept"] - expected_intercept) < 1e-4
        assert abs(result.estimates["x1"] - expected_x1) < 1e-4
        assert abs(result.estimates["x2"] - expected_x2) < 1e-4


# ============================================================================
# Tests Binomiale Négative
# ============================================================================


class TestNegativeBinomialRegression:
    def test_fit_negative_binomial(self, count_data):
        result = run_negative_binomial_regression(count_data, "y", ["x1", "x2"])
        assert result.method == "negative_binomial_regression"
        assert "Intercept" in result.estimates
        assert "alpha" in result.model_metrics
        assert result.model_metrics["alpha"] > 0  # Doit estimer une surdispersion
        assert "incidence_rate_ratios" in result.model_metrics

    def test_negative_binomial_vs_r(self, count_data):
        """Validation contre R : MASS::glm.nb(y ~ x1 + x2)"""
        result = run_negative_binomial_regression(count_data, "y", ["x1", "x2"])

        # Valeurs exactes extraites de R
        expected_intercept = 0.3907382
        expected_x1 = 0.3249544
        expected_x2 = 0.1394645

        assert abs(result.estimates["Intercept"] - expected_intercept) < 1e-4
        assert abs(result.estimates["x1"] - expected_x1) < 1e-4
        assert abs(result.estimates["x2"] - expected_x2) < 1e-4


# ============================================================================
# Tests Gamma
# ============================================================================


class TestGammaRegression:
    def test_fit_gamma(self, positive_data):
        result = run_gamma_regression(positive_data, "y", ["x1", "x2"], link="log")
        assert result.method == "gamma_regression"
        assert "Intercept" in result.estimates
        assert "scale" in result.model_metrics
        assert result.sample_size == 200

    def test_gamma_vs_r(self, positive_data):
        """Validation contre R : glm(y ~ x1 + x2, family=Gamma(link="log"))"""
        result = run_gamma_regression(positive_data, "y", ["x1", "x2"], link="log")

        # Valeurs exactes extraites de R
        expected_intercept = 0.9177360
        expected_x1 = 0.3268618
        expected_x2 = 0.3521254

        assert abs(result.estimates["Intercept"] - expected_intercept) < 1e-4
        assert abs(result.estimates["x1"] - expected_x1) < 1e-4
        assert abs(result.estimates["x2"] - expected_x2) < 1e-4