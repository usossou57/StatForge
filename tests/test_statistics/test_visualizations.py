"""Tests pour le module app/statistics/visualizations.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.results.models import AnalysisResult
from app.statistics.visualizations import (
    generate_bar_chart,
    generate_boxplot,
    generate_correlation_heatmap,
    generate_histogram,
    generate_scatter,
)


@pytest.fixture
def df_viz() -> pd.DataFrame:
    """DataFrame de test pour les visualisations."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "numeric_1": np.random.randn(n),
            "numeric_2": 0.5 * np.random.randn(n) + np.random.randn(n),
            "category": np.random.choice(["A", "B", "C"], size=n),
        }
    )


def test_generate_histogram(df_viz: pd.DataFrame) -> None:
    """Teste la génération d'un histogramme."""
    result = generate_histogram(df_viz, "numeric_1", kde=True)
    assert isinstance(result, AnalysisResult)
    assert result.method == "generate_histogram"
    assert "plots" in result.model_metrics
    assert "histogram" in result.model_metrics["plots"]
    assert result.model_metrics["plots"]["histogram"].startswith(
        "data:image/png;base64,"
    )


def test_generate_histogram_invalid_column(df_viz: pd.DataFrame) -> None:
    """Teste qu'une colonne manquante lève une ValueError."""
    with pytest.raises(ValueError, match="n'existe pas"):
        generate_histogram(df_viz, "inexistante")


def test_generate_boxplot(df_viz: pd.DataFrame) -> None:
    """Teste la génération d'un boxplot."""
    result = generate_boxplot(df_viz, x="category", y="numeric_1")
    assert isinstance(result, AnalysisResult)
    assert result.method == "generate_boxplot"
    assert "boxplot" in result.model_metrics["plots"]
    assert result.model_metrics["plots"]["boxplot"].startswith("data:image/png;base64,")


def test_generate_boxplot_invalid_types(df_viz: pd.DataFrame) -> None:
    """Teste qu'une variable Y non numérique lève une ValueError."""
    with pytest.raises(ValueError, match="doit être numérique"):
        generate_boxplot(df_viz, x="numeric_1", y="category")


def test_generate_scatter(df_viz: pd.DataFrame) -> None:
    """Teste la génération d'un scatter plot avec trendline."""
    result = generate_scatter(df_viz, x="numeric_1", y="numeric_2", trendline=True)
    assert isinstance(result, AnalysisResult)
    assert result.method == "generate_scatter"
    assert "scatter" in result.model_metrics["plots"]
    assert "correlation" in result.estimates


def test_generate_correlation_heatmap(df_viz: pd.DataFrame) -> None:
    """Teste la génération d'une heatmap de corrélation."""
    result = generate_correlation_heatmap(df_viz, variables=["numeric_1", "numeric_2"])
    assert isinstance(result, AnalysisResult)
    assert result.method == "generate_correlation_heatmap"
    assert "heatmap" in result.model_metrics["plots"]
    assert "correlation_matrix" in result.estimates


def test_generate_bar_chart(df_viz: pd.DataFrame) -> None:
    """Teste la génération d'un diagramme en barres."""
    result = generate_bar_chart(df_viz, column="category", normalize=False)
    assert isinstance(result, AnalysisResult)
    assert result.method == "generate_bar_chart"
    assert "bar_chart" in result.model_metrics["plots"]
    assert result.model_metrics["plots"]["bar_chart"].startswith(
        "data:image/png;base64,"
    )


def test_generate_bar_chart_normalized(df_viz: pd.DataFrame) -> None:
    """Teste la génération d'un diagramme en barres normalisé (pourcentages)."""
    result = generate_bar_chart(df_viz, column="category", normalize=True)
    assert isinstance(result, AnalysisResult)
    # Vérifie que le code généré contient normalize=True
    assert "normalize=True" in result.generated_code
