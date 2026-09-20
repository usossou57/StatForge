"""Tests pour app/statistics/engine_python.py."""

import pandas as pd
import pytest

from app.results.models import AnalysisResult
from app.statistics.engine_python import PythonStatisticalEngine


@pytest.fixture
def engine() -> PythonStatisticalEngine:
    """Fixture fournissant une instance du moteur Python."""
    return PythonStatisticalEngine()


@pytest.fixture
def df_ols() -> pd.DataFrame:
    """Dataset de test pour OLS."""
    return pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0]
    })


@pytest.fixture
def df_logistic() -> pd.DataFrame:
    """Dataset de test pour Logistic (sans séparation parfaite)."""
    return pd.DataFrame({
        "y": [0, 0, 0, 1, 1, 1],
        "x": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0]
    })


def test_engine_implements_protocol(engine: PythonStatisticalEngine) -> None:
    """Vérifie que PythonStatisticalEngine implémente bien le Protocol."""
    from app.statistics.provider import StatisticalEngineProvider
    assert isinstance(engine, StatisticalEngineProvider)


def test_run_linear_regression_returns_analysis_result(
    engine: PythonStatisticalEngine, df_ols: pd.DataFrame
) -> None:
    """Vérifie que run_linear_regression retourne un AnalysisResult valide."""
    result = engine.run_linear_regression(df_ols, "y", ["x"])
    
    assert isinstance(result, AnalysisResult)
    assert result.method == "ols"
    assert "x" in result.estimates
    assert "r_squared" in result.model_metrics


def test_run_logistic_regression_returns_analysis_result(
    engine: PythonStatisticalEngine, df_logistic: pd.DataFrame
) -> None:
    """Vérifie que run_logistic_regression retourne un AnalysisResult valide."""
    result = engine.run_logistic_regression(df_logistic, "y", ["x"])
    
    assert isinstance(result, AnalysisResult)
    assert result.method == "logistic"
    assert "x" in result.estimates
    assert "odds_ratios" in result.model_metrics
