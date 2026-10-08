"""Génération de visualisations pour l'Exploratory Data Analysis (EDA).

Ce module fournit des fonctions pour générer des graphiques statistiques
(histogrammes, boxplots, scatter plots, heatmaps) et les retourner encodés
en base64 pour un affichage facile dans une interface web.
"""
from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]

from app.results.models import AnalysisResult
from app.statistics.registry import MethodSpec, register_method


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convertit une figure matplotlib en chaîne base64 (format PNG)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def generate_histogram(
    df: pd.DataFrame, column: str, kde: bool = True, bins: int = 30
) -> AnalysisResult:
    """Génère un histogramme avec option de densité (KDE)."""
    if column not in df.columns:
        raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame.")
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"La colonne '{column}' doit être numérique pour un histogramme.")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[column].dropna(), bins=bins, kde=kde, ax=ax, color="skyblue")
    ax.set_title(f"Distribution de {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Fréquence")
    
    img_base64 = _fig_to_base64(fig)

    return AnalysisResult(
        method="generate_histogram",
        dataset_id="in_memory",
        variables={"column": column, "kde": kde, "bins": bins},
        sample_size=len(df[column].dropna()),
        preprocessing=["Suppression des valeurs manquantes"],
        estimates={},
        standard_errors={},
        confidence_intervals={},
        p_values={},
        model_metrics={
            "method_family": "visualization",
            "plots": {"histogram": img_base64},
        },
        diagnostics=[],
        warnings=[],
        assumptions_checked=[],
        generated_code=(
            f"import seaborn as sns\nimport matplotlib.pyplot as plt\n"
            f"sns.histplot(df['{column}'].dropna(), bins={bins}, kde={kde})\nplt.show()"
        ),
        execution_log=["Histogramme généré avec succès"],
        software_versions={"matplotlib": plt.matplotlib.__version__, "seaborn": sns.__version__},
    )


def generate_boxplot(
    df: pd.DataFrame, x: str, y: str
) -> AnalysisResult:
    """Génère un boxplot pour comparer la distribution d'une variable numérique par catégorie."""
    if x not in df.columns or y not in df.columns:
        raise ValueError("Les colonnes x et y doivent exister dans le DataFrame.")
    
    if not pd.api.types.is_numeric_dtype(df[y]):
        raise ValueError(f"La variable y ('{y}') doit être numérique.")

    fig, ax = plt.subplots(figsize=(8, 5))
    # Ajout de hue et legend=False pour éviter les warnings de dépréciation seaborn
    sns.boxplot(
        x=x, y=y, hue=x, data=df.dropna(subset=[x, y]), ax=ax, palette="Set2", legend=False
    )
    ax.set_title(f"Boxplot de {y} par {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.xticks(rotation=45)
    
    img_base64 = _fig_to_base64(fig)

    return AnalysisResult(
        method="generate_boxplot",
        dataset_id="in_memory",
        variables={"x": x, "y": y},
        sample_size=len(df.dropna(subset=[x, y])),
        preprocessing=["Suppression des lignes avec valeurs manquantes sur x ou y"],
        estimates={},
        standard_errors={},
        confidence_intervals={},
        p_values={},
        model_metrics={
            "method_family": "visualization",
            "plots": {"boxplot": img_base64},
        },
        diagnostics=[],
        warnings=[],
        assumptions_checked=[],
        generated_code=(
            f"import seaborn as sns\nimport matplotlib.pyplot as plt\n"
            f"sns.boxplot(x='{x}', y='{y}', hue='{x}', data=df, legend=False)\nplt.show()"
        ),
        execution_log=["Boxplot généré avec succès"],
        software_versions={"matplotlib": plt.matplotlib.__version__, "seaborn": sns.__version__},
    )


def generate_scatter(
    df: pd.DataFrame, x: str, y: str, trendline: bool = True
) -> AnalysisResult:
    """Génère un nuage de points (scatter plot) avec option de ligne de tendance."""
    if x not in df.columns or y not in df.columns:
        raise ValueError("Les colonnes x et y doivent exister dans le DataFrame.")
    
    if not pd.api.types.is_numeric_dtype(df[x]) or not pd.api.types.is_numeric_dtype(df[y]):
        raise ValueError("Les variables x et y doivent être numériques.")

    subset = df.dropna(subset=[x, y])
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if trendline:
        sns.regplot(x=x, y=y, data=subset, ax=ax, scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
    else:
        sns.scatterplot(x=x, y=y, data=subset, ax=ax, alpha=0.6)
        
    ax.set_title(f"Nuage de points : {y} en fonction de {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    
    img_base64 = _fig_to_base64(fig)

    # Calcul optionnel de la corrélation pour les métriques
    r, p = stats.pearsonr(subset[x], subset[y])

    return AnalysisResult(
        method="generate_scatter",
        dataset_id="in_memory",
        variables={"x": x, "y": y, "trendline": trendline},
        sample_size=len(subset),
        preprocessing=["Suppression des valeurs manquantes"],
        estimates={"correlation": float(r)},
        standard_errors={},
        confidence_intervals={},
        p_values={"correlation": float(p)},
        model_metrics={
            "method_family": "visualization",
            "plots": {"scatter": img_base64},
        },
        diagnostics=[],
        warnings=[],
        assumptions_checked=[],
        generated_code=(
            f"import seaborn as sns\nimport matplotlib.pyplot as plt\n"
            f"sns.regplot(x='{x}', y='{y}', data=df)\nplt.show()"
        ),
        execution_log=["Scatter plot généré avec succès"],
        software_versions={"matplotlib": plt.matplotlib.__version__, "seaborn": sns.__version__},
    )


def generate_correlation_heatmap(
    df: pd.DataFrame, variables: list[str], method: str = "pearson"
) -> AnalysisResult:
    """Génère une heatmap de la matrice de corrélation."""
    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    subset = df[variables].dropna()
    corr_matrix = subset.corr(method=method)  # type: ignore[arg-type]

    fig, ax = plt.subplots(figsize=(max(6, len(variables)), max(5, len(variables))))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=-1, 
        vmax=1, 
        ax=ax,
        square=True
    )
    ax.set_title(f"Matrice de corrélation ({method.capitalize()})")
    
    img_base64 = _fig_to_base64(fig)

    return AnalysisResult(
        method="generate_correlation_heatmap",
        dataset_id="in_memory",
        variables={"columns": variables, "method": method},
        sample_size=len(subset),
        preprocessing=["Suppression des lignes avec valeurs manquantes"],
        estimates={"correlation_matrix": corr_matrix.to_dict()},
        standard_errors={},
        confidence_intervals={},
        p_values={},
        model_metrics={
            "method_family": "visualization",
            "plots": {"heatmap": img_base64},
        },
        diagnostics=[],
        warnings=[],
        assumptions_checked=[],
        generated_code=(
            f"import seaborn as sns\nimport matplotlib.pyplot as plt\n"
            f"corr = df[{variables}].corr(method='{method}')\n"
            f"sns.heatmap(corr, annot=True, cmap='coolwarm')\nplt.show()"
        ),
        execution_log=["Heatmap de corrélation générée avec succès"],
        software_versions={"matplotlib": plt.matplotlib.__version__, "seaborn": sns.__version__},
    )


def generate_bar_chart(
    df: pd.DataFrame, column: str, normalize: bool = False
) -> AnalysisResult:
    """Génère un diagramme en barres pour une variable catégorielle."""
    if column not in df.columns:
        raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame.")

    # Assignation unique avec cast explicite en float pour satisfaire mypy
    counts = (
        df[column].value_counts(normalize=True) * 100.0
        if normalize
        else df[column].value_counts(normalize=False).astype(float)
    )
    ylabel = "Pourcentage (%)" if normalize else "Effectif"

    fig, ax = plt.subplots(figsize=(8, 5))
    # Ajout de hue et legend=False pour éviter les warnings de dépréciation seaborn
    sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        hue=counts.index.astype(str),
        ax=ax,
        palette="viridis",
        legend=False,
    )
    ax.set_title(f"Distribution de {column}")
    ax.set_xlabel(column)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45)
    
    img_base64 = _fig_to_base64(fig)

    return AnalysisResult(
        method="generate_bar_chart",
        dataset_id="in_memory",
        variables={"column": column, "normalize": normalize},
        sample_size=len(df[column].dropna()),
        preprocessing=["Calcul des effectifs ou pourcentages"],
        estimates={},
        standard_errors={},
        confidence_intervals={},
        p_values={},
        model_metrics={
            "method_family": "visualization",
            "plots": {"bar_chart": img_base64},
        },
        diagnostics=[],
        warnings=[],
        assumptions_checked=[],
        generated_code=(
            f"import seaborn as sns\nimport matplotlib.pyplot as plt\n"
            f"counts = df['{column}'].value_counts(normalize={normalize})\n"
            f"sns.barplot(x=counts.index.astype(str), y=counts.values, hue=counts.index.astype(str), legend=False)\nplt.show()"
        ),
        execution_log=["Diagramme en barres généré avec succès"],
        software_versions={"matplotlib": plt.matplotlib.__version__, "seaborn": sns.__version__},
    )


# ==============================================================================
# Enregistrement dans le registre (V2 §7.1)
# ==============================================================================
register_method(
    MethodSpec(
        method_id="generate_histogram",
        label_fr="Histogramme avec KDE",
        family="visualization",
        engine="python",
        outcome_types=["numeric"],
        predictor_types=[],
        min_sample_size=10,
        assumptions=["Variable continue ou discrète avec suffisamment de valeurs"],
        reference_impl="seaborn.histplot",
        r_reference="geom_histogram",
        phase=16,
    ),
    generate_histogram,
)

register_method(
    MethodSpec(
        method_id="generate_boxplot",
        label_fr="Boîte à moustaches (Boxplot)",
        family="visualization",
        engine="python",
        outcome_types=["numeric"],
        predictor_types=["categorical"],
        min_sample_size=5,
        assumptions=["Variable Y numérique, Variable X catégorielle"],
        reference_impl="seaborn.boxplot",
        r_reference="geom_boxplot",
        phase=16,
    ),
    generate_boxplot,
)

register_method(
    MethodSpec(
        method_id="generate_scatter",
        label_fr="Nuage de points (Scatter plot)",
        family="visualization",
        engine="python",
        outcome_types=["numeric"],
        predictor_types=["numeric"],
        min_sample_size=5,
        assumptions=["Relation linéaire potentielle"],
        reference_impl="seaborn.regplot",
        r_reference="geom_point + geom_smooth",
        phase=16,
    ),
    generate_scatter,
)

register_method(
    MethodSpec(
        method_id="generate_correlation_heatmap",
        label_fr="Heatmap de corrélation",
        family="visualization",
        engine="python",
        outcome_types=["numeric"],
        predictor_types=["numeric"],
        min_sample_size=5,
        assumptions=["Variables numériques"],
        reference_impl="seaborn.heatmap",
        r_reference="corrplot",
        phase=16,
    ),
    generate_correlation_heatmap,
)

register_method(
    MethodSpec(
        method_id="generate_bar_chart",
        label_fr="Diagramme en barres",
        family="visualization",
        engine="python",
        outcome_types=["categorical", "ordinal"],
        predictor_types=[],
        min_sample_size=1,
        assumptions=["Variable catégorielle"],
        reference_impl="seaborn.barplot",
        r_reference="geom_bar",
        phase=16,
    ),
    generate_bar_chart,
)