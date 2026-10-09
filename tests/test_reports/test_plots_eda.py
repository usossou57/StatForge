"""Tests pour le module app/reports/plots_eda.py.

Validation des fonctions de visualisation exploratoire.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from app.reports.plots_eda import (
    plot_boxplot,
    plot_correlation_heatmap,
    plot_correlation_heatmap_interactive,
    plot_histogram,
    plot_pairs,
    plot_qq,
    plot_scatter,
    plot_scatter_interactive,
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


@pytest.fixture
def df_large() -> pd.DataFrame:
    """DataFrame volumineux pour tester l'échantillonnage."""
    np.random.seed(42)
    n = 10000
    return pd.DataFrame(
        {
            "x": np.random.randn(n),
            "y": np.random.randn(n),
        }
    )


# ==============================================================================
# Tests pour plot_histogram
# ==============================================================================


def test_plot_histogram_basic(df_viz: pd.DataFrame) -> None:
    """Teste la génération d'un histogramme basique."""
    fig = plot_histogram(df_viz, "numeric_1")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_histogram_without_kde(df_viz: pd.DataFrame) -> None:
    """Teste un histogramme sans KDE."""
    fig = plot_histogram(df_viz, "numeric_1", kde=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_histogram_invalid_column(df_viz: pd.DataFrame) -> None:
    """Teste qu'une colonne manquante lève ValueError."""
    with pytest.raises(ValueError, match="n'existe pas"):
        plot_histogram(df_viz, "inexistante")


def test_plot_histogram_non_numeric(df_viz: pd.DataFrame) -> None:
    """Teste qu'une colonne non numérique lève ValueError."""
    with pytest.raises(ValueError, match="doit être numérique"):
        plot_histogram(df_viz, "category")


# ==============================================================================
# Tests pour plot_boxplot
# ==============================================================================


def test_plot_boxplot_basic(df_viz: pd.DataFrame) -> None:
    """Teste la génération d'un boxplot."""
    fig = plot_boxplot(df_viz, x="category", y="numeric_1")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_boxplot_invalid_y(df_viz: pd.DataFrame) -> None:
    """Teste qu'une variable Y non numérique lève ValueError."""
    with pytest.raises(ValueError, match="doit être numérique"):
        plot_boxplot(df_viz, x="numeric_1", y="category")


# ==============================================================================
# Tests pour plot_qq
# ==============================================================================


def test_plot_qq_basic(df_viz: pd.DataFrame) -> None:
    """Teste la génération d'un Q-Q plot."""
    fig = plot_qq(df_viz, "numeric_1")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_qq_non_numeric(df_viz: pd.DataFrame) -> None:
    """Teste qu'une colonne non numérique lève ValueError."""
    with pytest.raises(ValueError, match="doit être numérique"):
        plot_qq(df_viz, "category")


# ==============================================================================
# Tests pour plot_scatter
# ==============================================================================


def test_plot_scatter_with_trendline(df_viz: pd.DataFrame) -> None:
    """Teste un scatter plot avec ligne de tendance."""
    fig = plot_scatter(df_viz, x="numeric_1", y="numeric_2", trendline=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_scatter_without_trendline(df_viz: pd.DataFrame) -> None:
    """Teste un scatter plot sans ligne de tendance."""
    fig = plot_scatter(df_viz, x="numeric_1", y="numeric_2", trendline=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_scatter_non_numeric(df_viz: pd.DataFrame) -> None:
    """Teste qu'une variable non numérique lève ValueError."""
    with pytest.raises(ValueError, match="doivent être numériques"):
        plot_scatter(df_viz, x="numeric_1", y="category")


# ==============================================================================
# Tests pour plot_correlation_heatmap
# ==============================================================================


def test_plot_correlation_heatmap_basic(df_viz: pd.DataFrame) -> None:
    """Teste la génération d'une heatmap de corrélation."""
    fig = plot_correlation_heatmap(df_viz, variables=["numeric_1", "numeric_2"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_correlation_heatmap_invalid_method(df_viz: pd.DataFrame) -> None:
    """Teste qu'une méthode invalide lève ValueError."""
    with pytest.raises(ValueError, match="non reconnue"):
        plot_correlation_heatmap(df_viz, variables=["numeric_1"], method="invalid")


def test_plot_correlation_heatmap_missing_column(df_viz: pd.DataFrame) -> None:
    """Teste qu'une colonne manquante lève ValueError."""
    with pytest.raises(ValueError, match="manquantes"):
        plot_correlation_heatmap(df_viz, variables=["numeric_1", "inexistante"])


# ==============================================================================
# Tests pour plot_pairs
# ==============================================================================


def test_plot_pairs_basic(df_viz: pd.DataFrame) -> None:
    """Teste la génération d'une matrice de nuages de points."""
    fig = plot_pairs(df_viz, variables=["numeric_1", "numeric_2"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_pairs_max_vars(df_viz: pd.DataFrame) -> None:
    """Teste que plot_pairs limite le nombre de variables."""
    # Ajouter plus de colonnes numériques
    df_viz["num_3"] = np.random.randn(len(df_viz))
    df_viz["num_4"] = np.random.randn(len(df_viz))
    df_viz["num_5"] = np.random.randn(len(df_viz))
    df_viz["num_6"] = np.random.randn(len(df_viz))
    df_viz["num_7"] = np.random.randn(len(df_viz))

    fig = plot_pairs(
        df_viz,
        variables=[
            "numeric_1",
            "numeric_2",
            "num_3",
            "num_4",
            "num_5",
            "num_6",
            "num_7",
        ],
        max_vars=4,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ==============================================================================
# Tests pour l'échantillonnage
# ==============================================================================


def test_sampling_large_dataset(df_large: pd.DataFrame) -> None:
    """Teste que les fonctions échantillonnent les gros datasets."""
    # plot_scatter doit échantillonner car df_large a 10000 lignes
    fig = plot_scatter(df_large, x="x", y="y")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ==============================================================================
# Tests pour les variantes interactives Plotly (optionnels)
# ==============================================================================


def test_plot_scatter_interactive(df_viz: pd.DataFrame) -> None:
    """Teste la génération d'un scatter plot interactif (si Plotly installé)."""
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]

        fig = plot_scatter_interactive(df_viz, x="numeric_1", y="numeric_2")
        assert isinstance(fig, go.Figure)
    except ImportError:
        pytest.skip("Plotly non installé")


def test_plot_correlation_heatmap_interactive(df_viz: pd.DataFrame) -> None:
    """Teste la génération d'une heatmap interactive (si Plotly installé)."""
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]

        fig = plot_correlation_heatmap_interactive(
            df_viz, variables=["numeric_1", "numeric_2"]
        )
        assert isinstance(fig, go.Figure)
    except ImportError:
        pytest.skip("Plotly non installé")
