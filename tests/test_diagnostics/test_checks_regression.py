"""Tests pour les diagnostics de régression."""
import numpy as np
import pandas as pd
import pytest

from app.diagnostics.checks_regression import run_regression_diagnostics


@pytest.fixture
def regression_data():
    """Données de régression pour tests de diagnostics."""
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    
    # Relation linéaire + bruit
    y = 2.0 * x1 + 1.5 * x2 + np.random.randn(n) * 0.5
    
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    return df


class TestRegressionDiagnostics:
    def test_basic_diagnostics(self, regression_data):
        """Test de base des diagnostics."""
        result = run_regression_diagnostics(
            regression_data, "y", ["x1", "x2"], model_type="ols"
        )
        
        assert result.method == "ols_regression_diagnostics"
        assert len(result.residuals) == 100
        assert len(result.fitted_values) == 100
        assert "shapiro_wilk" in result.normality_tests
        assert "breusch_pagan" in result.homoscedasticity_tests
        assert "durbin_watson" in result.autocorrelation_tests
        assert len(result.vif) == 2  # x1 et x2
        assert len(result.cooks_distance) == 100
        assert len(result.leverage) == 100

    def test_normality_tests(self, regression_data):
        """Test des tests de normalité."""
        result = run_regression_diagnostics(
            regression_data, "y", ["x1", "x2"], model_type="ols"
        )
        
        # Shapiro-Wilk
        assert "statistic" in result.normality_tests["shapiro_wilk"]
        assert "p_value" in result.normality_tests["shapiro_wilk"]
        assert "normal" in result.normality_tests["shapiro_wilk"]
        
        # Jarque-Bera
        assert "statistic" in result.normality_tests["jarque_bera"]
        assert "p_value" in result.normality_tests["jarque_bera"]

    def test_homoscedasticity_tests(self, regression_data):
        """Test des tests d'homoscédasticité."""
        result = run_regression_diagnostics(
            regression_data, "y", ["x1", "x2"], model_type="ols"
        )
        
        # Breusch-Pagan
        assert "statistic" in result.homoscedasticity_tests["breusch_pagan"]
        assert "p_value" in result.homoscedasticity_tests["breusch_pagan"]
        assert "homoscedastic" in result.homoscedasticity_tests["breusch_pagan"]

    def test_vif_calculation(self, regression_data):
        """Test du calcul du VIF."""
        result = run_regression_diagnostics(
            regression_data, "y", ["x1", "x2"], model_type="ols"
        )
        
        assert "x1" in result.vif
        assert "x2" in result.vif
        assert result.vif["x1"] > 0
        assert result.vif["x2"] > 0

    def test_plots_generation(self, regression_data, tmp_path):
        """Test de la génération des graphiques."""
        output_dir = str(tmp_path / "diagnostics")
        result = run_regression_diagnostics(
            regression_data, "y", ["x1", "x2"], 
            model_type="ols", 
            output_dir=output_dir
        )
        
        assert "residuals_vs_fitted" in result.plots
        assert "qq_plot" in result.plots
        assert "scale_location" in result.plots
        assert "residuals_vs_leverage" in result.plots
        
        # Vérifier que les fichiers existent
        import os
        for plot_path in result.plots.values():
            assert os.path.exists(plot_path)

    def test_generated_code(self, regression_data):
        """Test que le code reproductible est généré."""
        result = run_regression_diagnostics(
            regression_data, "y", ["x1", "x2"], model_type="ols"
        )
        
        assert len(result.generated_code) > 0
        assert "import pandas as pd" in result.generated_code
        assert "import statsmodels.api as sm" in result.generated_code

    def test_assumptions_interpretation(self, regression_data):
        """Test de l'interprétation des hypothèses."""
        result = run_regression_diagnostics(
            regression_data, "y", ["x1", "x2"], model_type="ols"
        )
        
        # Vérifier que les hypothèses sont évaluées
        assert "normality" in result.assumptions_met
        assert "homoscedasticity" in result.assumptions_met
        assert "no_autocorrelation" in result.assumptions_met
        assert "no_multicollinearity" in result.assumptions_met
        assert "no_influential_points" in result.assumptions_met