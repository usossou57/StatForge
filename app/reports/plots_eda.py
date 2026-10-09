"""Graphiques exploratoires pour l'EDA (Exploratory Data Analysis).

Ce module fournit des fonctions de visualisation qui retournent des objets
matplotlib.figure.Figure pour une intégration facile dans des rapports HTML,
des notebooks ou des interfaces web.

Conforme à la spécification V2 §9 Phase 16 T16.4.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]


def _sample_if_needed(
    df: pd.DataFrame, max_rows: int = 5000, random_state: int = 0
) -> pd.DataFrame:
    """Échantillonne le DataFrame si nécessaire pour les graphiques."""
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=random_state)
    return df


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    kde: bool = True,
    bins: int = 30,
    title: str | None = None,
) -> plt.Figure:
    """Génère un histogramme avec option de densité (KDE).

    Args:
        df: DataFrame contenant les données.
        column: Nom de la colonne numérique.
        kde: Afficher la courbe de densité (Kernel Density Estimation).
        bins: Nombre de bins pour l'histogramme.
        title: Titre du graphique (par défaut : "Distribution de {column}").

    Returns:
        matplotlib.figure.Figure contenant l'histogramme.

    Raises:
        ValueError: Si la colonne n'existe pas ou n'est pas numérique.
    """
    if column not in df.columns:
        raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"La colonne '{column}' doit être numérique pour un histogramme."
        )

    df_sampled = _sample_if_needed(df[[column]].dropna())

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_sampled[column], bins=bins, kde=kde, ax=ax, color="skyblue")
    ax.set_title(title or f"Distribution de {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Fréquence")

    return fig


def plot_boxplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str | None = None,
) -> plt.Figure:
    """Génère un boxplot pour comparer la distribution d'une variable numérique par catégorie.

    Args:
        df: DataFrame contenant les données.
        x: Nom de la variable catégorielle (axes X).
        y: Nom de la variable numérique (axes Y).
        title: Titre du graphique.

    Returns:
        matplotlib.figure.Figure contenant le boxplot.

    Raises:
        ValueError: Si les colonnes n'existent pas ou si y n'est pas numérique.
    """
    if x not in df.columns or y not in df.columns:
        raise ValueError("Les colonnes x et y doivent exister dans le DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[y]):
        raise ValueError(f"La variable y ('{y}') doit être numérique.")

    df_sampled = _sample_if_needed(df[[x, y]].dropna())

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=x, y=y, hue=x, data=df_sampled, ax=ax, palette="Set2", legend=False)
    ax.set_title(title or f"Boxplot de {y} par {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.xticks(rotation=45)

    return fig


def plot_qq(
    df: pd.DataFrame,
    column: str,
    title: str | None = None,
) -> plt.Figure:
    """Génère un Q-Q plot (Quantile-Quantile) pour vérifier la normalité.

    Args:
        df: DataFrame contenant les données.
        column: Nom de la colonne numérique.
        title: Titre du graphique.

    Returns:
        matplotlib.figure.Figure contenant le Q-Q plot.

    Raises:
        ValueError: Si la colonne n'existe pas ou n'est pas numérique.
    """
    if column not in df.columns:
        raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"La colonne '{column}' doit être numérique pour un Q-Q plot.")

    values = df[column].dropna().to_numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    stats.probplot(values, dist="norm", plot=ax)
    ax.set_title(title or f"Q-Q Plot de {column}")

    return fig


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    trendline: bool = True,
    title: str | None = None,
) -> plt.Figure:
    """Génère un nuage de points (scatter plot) avec option de ligne de tendance.

    Args:
        df: DataFrame contenant les données.
        x: Nom de la variable numérique pour l'axe X.
        y: Nom de la variable numérique pour l'axe Y.
        trendline: Afficher une ligne de tendance (régression linéaire).
        title: Titre du graphique.

    Returns:
        matplotlib.figure.Figure contenant le scatter plot.

    Raises:
        ValueError: Si les colonnes n'existent pas ou ne sont pas numériques.
    """
    if x not in df.columns or y not in df.columns:
        raise ValueError("Les colonnes x et y doivent exister dans le DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[x]) or not pd.api.types.is_numeric_dtype(
        df[y]
    ):
        raise ValueError("Les variables x et y doivent être numériques.")

    df_sampled = _sample_if_needed(df[[x, y]].dropna())

    fig, ax = plt.subplots(figsize=(8, 5))

    if trendline:
        sns.regplot(
            x=x,
            y=y,
            data=df_sampled,
            ax=ax,
            scatter_kws={"alpha": 0.6},
            line_kws={"color": "red"},
        )
    else:
        sns.scatterplot(x=x, y=y, data=df_sampled, ax=ax, alpha=0.6)

    ax.set_title(title or f"Nuage de points : {y} en fonction de {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    variables: list[str],
    method: str = "pearson",
    title: str | None = None,
) -> plt.Figure:
    """Génère une heatmap de la matrice de corrélation.

    Args:
        df: DataFrame contenant les données.
        variables: Liste des colonnes numériques à corréler.
        method: Méthode de corrélation ("pearson", "spearman", "kendall").
        title: Titre du graphique.

    Returns:
        matplotlib.figure.Figure contenant la heatmap.

    Raises:
        ValueError: Si des colonnes manquent ou si la méthode est invalide.
    """
    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    valid_methods = ("pearson", "spearman", "kendall")
    if method not in valid_methods:
        raise ValueError(
            f"Méthode '{method}' non reconnue. Choisir parmi {valid_methods}."
        )

    df_sampled = _sample_if_needed(df[variables].dropna())
    corr_matrix = df_sampled.corr(method=method)  # type: ignore[arg-type]

    fig, ax = plt.subplots(figsize=(max(6, len(variables)), max(5, len(variables))))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
    )
    ax.set_title(title or f"Matrice de corrélation ({method.capitalize()})")

    return fig


def plot_pairs(
    df: pd.DataFrame,
    variables: list[str],
    max_vars: int = 6,
    title: str | None = None,
) -> plt.Figure:
    """Génère une matrice de nuages de points (pair plot) pour explorer les relations.

    Args:
        df: DataFrame contenant les données.
        variables: Liste des colonnes numériques à inclure.
        max_vars: Nombre maximum de variables à afficher (pour éviter les graphiques trop lourds).
        title: Titre du graphique (super title).

    Returns:
        matplotlib.figure.Figure contenant la matrice de nuages de points.

    Raises:
        ValueError: Si des colonnes manquent.
    """
    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    # Limiter le nombre de variables pour éviter les graphiques trop lourds
    if len(variables) > max_vars:
        variables = variables[:max_vars]

    df_sampled = _sample_if_needed(df[variables].dropna())

    g = sns.pairplot(df_sampled, diag_kind="kde", plot_kws={"alpha": 0.6})
    if title:
        g.fig.suptitle(title, y=1.02)

    return g.fig


# ==============================================================================
# Variantes interactives Plotly (optionnelles)
# ==============================================================================


def plot_scatter_interactive(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    title: str | None = None,
) -> Any:
    """Génère un nuage de points interactif avec Plotly.

    Args:
        df: DataFrame contenant les données.
        x: Nom de la variable numérique pour l'axe X.
        y: Nom de la variable numérique pour l'axe Y.
        color: Nom de la variable pour colorer les points (optionnel).
        title: Titre du graphique.

    Returns:
        plotly.graph_objects.Figure contenant le scatter plot interactif.

    Raises:
        ValueError: Si les colonnes n'existent pas.
    """
    try:
        import plotly.express as px  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "Plotly est requis pour les graphiques interactifs. "
            "Installez-le avec : uv add plotly"
        ) from e

    if x not in df.columns or y not in df.columns:
        raise ValueError("Les colonnes x et y doivent exister dans le DataFrame.")

    df_sampled = _sample_if_needed(df)

    fig = px.scatter(
        df_sampled,
        x=x,
        y=y,
        color=color,
        title=title or f"Nuage de points interactif : {y} en fonction de {x}",
        trendline="ols"
        if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y])
        else None,
    )

    return fig


def plot_correlation_heatmap_interactive(
    df: pd.DataFrame,
    variables: list[str],
    method: str = "pearson",
    title: str | None = None,
) -> Any:
    """Génère une heatmap interactive de la matrice de corrélation avec Plotly.

    Args:
        df: DataFrame contenant les données.
        variables: Liste des colonnes numériques à corréler.
        method: Méthode de corrélation ("pearson", "spearman", "kendall").
        title: Titre du graphique.

    Returns:
        plotly.graph_objects.Figure contenant la heatmap interactive.

    Raises:
        ValueError: Si des colonnes manquent ou si la méthode est invalide.
    """
    try:
        import plotly.express as px  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "Plotly est requis pour les graphiques interactifs. "
            "Installez-le avec : uv add plotly"
        ) from e

    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    valid_methods = ("pearson", "spearman", "kendall")
    if method not in valid_methods:
        raise ValueError(
            f"Méthode '{method}' non reconnue. Choisir parmi {valid_methods}."
        )

    df_sampled = _sample_if_needed(df[variables].dropna())
    corr_matrix = df_sampled.corr(method=method)  # type: ignore[arg-type]

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=title or f"Matrice de corrélation interactive ({method.capitalize()})",
    )

    return fig
