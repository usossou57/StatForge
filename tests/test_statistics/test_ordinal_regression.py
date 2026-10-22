"""Tests pour la régression logistique ordinale.
Inclut validation contre R (V2 §6.3, T12).

NOTE : Seuls les COEFFICIENTS sont validés contre R. Les seuils (cut-points)
ne sont pas comparables brute-à-brute entre statsmodels et MASS::polr
(paramétrisations différentes). Tolérance rel=1e-4 justifiée : optimisation itérative.
"""
import numpy as np
import pandas as pd
import pytest

from app.statistics.regression_glm import run_ordinal_regression


# ============================================================================
# Données de test (graine fixe pour reproductibilité)
# ============================================================================


@pytest.fixture
def ordinal_data():
    """Données ordinales pour test."""
    np.random.seed(42)
    n = 300
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)

    latent = 0.5 * x1 + 0.3 * x2 + np.random.randn(n)
    thresholds = [-1.0, 0.0, 1.0]
    y = np.digitize(latent, thresholds) + 1

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    df["y"] = pd.Categorical(df["y"], categories=[1, 2, 3, 4], ordered=True)

    df.to_csv("validation/data_ordinal_python.csv", index=False)
    return df


# ============================================================================
# Tests Régression Ordinale
# ============================================================================


class TestOrdinalRegression:
    def test_fit_ordinal(self, ordinal_data):
        result = run_ordinal_regression(ordinal_data, "y", ["x1", "x2"], link="logit")
        assert result.method == "ordinal_regression"
        assert "x1" in result.estimates
        assert "x2" in result.estimates
        assert "thresholds" in result.model_metrics
        assert result.model_metrics["n_categories"] == 4
        assert result.sample_size == 300
        assert "aic" in result.model_metrics
        # Vérifier que les seuils ont bien été extraits séparément
        assert len(result.model_metrics["thresholds"]) == 3

    def test_ordinal_vs_r_logit(self, ordinal_data):
        """Validation contre R : MASS::polr(y ~ x1 + x2, method="logistic")
        
        Seuls les coefficients sont comparés (tolérance rel=1e-4, V2 §6.3 : 
        optimisation itérative). Les seuils ne sont pas comparables.
        """
        result = run_ordinal_regression(ordinal_data, "y", ["x1", "x2"], link="logit")

        # Valeurs exactes extraites de R (r_validation_ordinal.R)
        expected_x1 = 0.7448664
        expected_x2 = 0.4706506

        # Diagnostic confirmé : statsmodels donne x1=0.744883, x2=0.470654 → concordance < 1e-4 ✅
        assert abs(result.estimates["x1"] - expected_x1) < 1e-4
        assert abs(result.estimates["x2"] - expected_x2) < 1e-4

    def test_ordinal_vs_r_probit(self, ordinal_data):
        """Validation contre R : MASS::polr(y ~ x1 + x2, method="probit")"""
        result = run_ordinal_regression(ordinal_data, "y", ["x1", "x2"], link="probit")

        # Valeurs exactes extraites de R
        expected_x1 = 0.4488531
        expected_x2 = 0.2825007

        assert abs(result.estimates["x1"] - expected_x1) < 1e-4
        assert abs(result.estimates["x2"] - expected_x2) < 1e-4