"""Génération de rapports (HTML/Markdown) et de graphiques pour les analyses.

Ce module produit des rapports structurés et des visualisations à partir
des objets AnalysisResult.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from app.data.models import DataDictionary

if TYPE_CHECKING:
    import matplotlib.figure

    from app.results.models import AnalysisResult


# Chemin vers les templates Jinja2
_TEMPLATES_DIR = Path(__file__).parent / "templates"


def generate_report(result: AnalysisResult, output_format: str = "html") -> str:
    """Génère un rapport d'analyse au format demandé.

    Args:
        result: Objet AnalysisResult contenant les résultats de l'analyse.
        output_format: Format de sortie, 'html' ou 'markdown'. Défaut 'html'.

    Returns:
        Chaîne de caractères contenant le rapport complet au format demandé.

    Raises:
        ValueError: Si output_format n'est pas 'html' ou 'markdown'.
    """
    if output_format not in ("html", "markdown"):
        raise ValueError(
            f"Format '{output_format}' non supporté. "
            "Formats disponibles : 'html', 'markdown'."
        )

    if output_format == "html":
        return _generate_html_report(result)
    else:
        return _generate_markdown_report(result)


def _generate_html_report(result: AnalysisResult) -> str:
    """Génère un rapport HTML via Jinja2.

    Args:
        result: Objet AnalysisResult.

    Returns:
        Chaîne HTML complète.
    """
    env = Environment(
        loader=FileSystemLoader(_TEMPLATES_DIR),
        autoescape=True,
    )
    template = env.get_template("report.html.j2")

    # Préparation des données pour le template
    coefficients_data = _prepare_coefficients_table(result)
    diagnostics_data = [
        {
            "code": d.code,
            "severity": d.severity.value,
            "message": d.message,
            "details": d.details,
        }
        for d in result.diagnostics
    ]

    return template.render(
        method=result.method,
        dataset_id=result.dataset_id,
        sample_size=result.sample_size,
        variables=result.variables,
        preprocessing=result.preprocessing,
        coefficients=coefficients_data,
        diagnostics=diagnostics_data,
        warnings=result.warnings,
        assumptions_checked=result.assumptions_checked,
        generated_code=result.generated_code,
        software_versions=result.software_versions,
        created_at=result.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
    )


def _generate_markdown_report(result: AnalysisResult) -> str:
    """Génère un rapport Markdown.

    Args:
        result: Objet AnalysisResult.

    Returns:
        Chaîne Markdown complète.
    """
    lines = [
        f"# Rapport d'analyse — {result.method}",
        "",
        f"**Dataset :** {result.dataset_id}",
        f"**Taille d'échantillon :** {result.sample_size}",
        f"**Date :** {result.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Variables",
        "",
    ]

    # Variables
    for var_name, var_info in result.variables.items():
        role = var_info.get("role", "N/A")
        lines.append(f"- **{var_name}** : {role}")
    lines.append("")

    # Prétraitement
    if result.preprocessing:
        lines.append("## Prétraitement")
        lines.append("")
        for step in result.preprocessing:
            lines.append(f"- {step}")
        lines.append("")

    # Tableau des coefficients
    coefficients_data = _prepare_coefficients_table(result)
    if coefficients_data:
        lines.extend(
            [
                "## Coefficients",
                "",
                "| Variable | Estimate | SE | IC 95% | p-value |",
                "|----------|----------|----|----|---------|",
            ]
        )
        for coef in coefficients_data:
            lines.append(
                f"| {coef['name']} | {coef['estimate']:.4f} | "
                f"{coef['se']:.4f} | [{coef['ci_lower']:.4f}, {coef['ci_upper']:.4f}] | "
                f"{coef['p_value']:.4e} |"
            )
        lines.append("")

    # Métriques du modèle
    if result.model_metrics:
        lines.append("## Métriques du modèle")
        lines.append("")
        for metric, value in result.model_metrics.items():
            lines.append(f"- **{metric}** : {value:.4f}")
        lines.append("")

    # Diagnostics
    if result.diagnostics:
        lines.append("## Diagnostics")
        lines.append("")
        for diag in result.diagnostics:
            lines.append(f"- **[{diag.severity.value}]** {diag.code} : {diag.message}")
        lines.append("")

    # Warnings
    if result.warnings:
        lines.append("## Avertissements")
        lines.append("")
        for warning in result.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    # Hypothèses vérifiées
    if result.assumptions_checked:
        lines.append("## Hypothèses vérifiées")
        lines.append("")
        for assumption in result.assumptions_checked:
            lines.append(f"- {assumption}")
        lines.append("")

    # Code reproductible
    lines.extend(
        [
            "## Code reproductible",
            "",
            "```python",
            result.generated_code,
            "```",
            "",
        ]
    )

    # Versions logicielles
    lines.append("## Versions logicielles")
    lines.append("")
    for software, version in result.software_versions.items():
        lines.append(f"- **{software}** : {version}")
    lines.append("")

    return "\n".join(lines)


def _prepare_coefficients_table(result: AnalysisResult) -> list[dict]:
    """Prépare les données pour le tableau des coefficients.

    Args:
        result: Objet AnalysisResult.

    Returns:
        Liste de dictionnaires avec les données des coefficients.
    """
    coefficients = []
    for var_name in result.estimates:
        estimate = result.estimates[var_name]
        se = result.standard_errors.get(var_name, 0.0)
        p_value = result.p_values.get(var_name, 1.0)

        # Intervalles de confiance
        ci = result.confidence_intervals.get(var_name, (0.0, 0.0))
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            ci_lower, ci_upper = ci
        else:
            ci_lower, ci_upper = 0.0, 0.0

        coefficients.append(
            {
                "name": var_name,
                "estimate": estimate,
                "se": se,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "p_value": p_value,
            }
        )

    return coefficients


def generate_plot(result: AnalysisResult, kind: str) -> matplotlib.figure.Figure:
    """Génère un graphique d'analyse.

    Args:
        result: Objet AnalysisResult contenant les résultats.
        kind: Type de graphique, 'coefficients' (forest plot) ou 'residuals'.

    Returns:
        Objet matplotlib.figure.Figure contenant le graphique.

    Raises:
        ValueError: Si kind n'est pas supporté.
        NotImplementedError: Si le type de graphique n'est pas encore implémenté.
    """
    if kind == "coefficients":
        return _generate_coefficients_plot(result)
    elif kind == "residuals":
        raise NotImplementedError(
            "Le graphique de résidus n'est pas encore implémenté. "
            "Types disponibles : 'coefficients'."
        )
    else:
        raise ValueError(
            f"Type de graphique '{kind}' non supporté. "
            "Types disponibles : 'coefficients'."
        )


def _generate_coefficients_plot(result: AnalysisResult) -> matplotlib.figure.Figure:
    """Génère un forest plot des coefficients avec intervalles de confiance.

    Args:
        result: Objet AnalysisResult.

    Returns:
        Figure matplotlib avec le forest plot.
    """
    coefficients_data = _prepare_coefficients_table(result)

    if not coefficients_data:
        # Si pas de coefficients, retourne une figure vide avec message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "Aucun coefficient disponible pour ce type d'analyse",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_axis_off()
        return fig

    # Préparation des données
    names = [c["name"] for c in coefficients_data]
    estimates = [c["estimate"] for c in coefficients_data]
    ci_lowers = [c["ci_lower"] for c in coefficients_data]
    ci_uppers = [c["ci_upper"] for c in coefficients_data]

    # Calcul des erreurs (distance entre estimate et bornes IC)
    xerr_lower = [est - ci_low for est, ci_low in zip(estimates, ci_lowers)]
    xerr_upper = [ci_up - est for est, ci_up in zip(estimates, ci_uppers)]

    # Création de la figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.5)))

    y_pos = range(len(names))

    # Forest plot
    ax.errorbar(
        estimates,
        y_pos,
        xerr=[xerr_lower, xerr_upper],
        fmt="o",
        color="black",
        ecolor="gray",
        capsize=5,
        markersize=8,
    )

    # Ligne verticale à x=0
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1, alpha=0.7)

    # Configuration des axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Estimate", fontsize=12)
    ax.set_title(
        f"Coefficients — {result.method}",
        fontsize=14,
        fontweight="bold",
    )
    ax.invert_yaxis()  # Premier coefficient en haut
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()

    return fig


def generate_eda_report(
    df: pd.DataFrame,
    data_dictionary: DataDictionary,
    output_format: str = "html",
) -> str:
    """Génère un rapport EDA (Exploratory Data Analysis) complet."""
    if output_format not in ("html", "markdown"):
        raise ValueError(
            f"Format '{output_format}' non supporté. "
            "Formats disponibles : 'html', 'markdown'."
        )

    # Importation locale pour éviter les dépendances circulaires
    from app.reports.plots_eda import (
        plot_boxplot,
        plot_correlation_heatmap,
        plot_histogram,
    )
    from app.statistics.correlation import run_correlation
    from app.statistics.descriptive import summarize_variables
    from app.statistics.normality_outliers import detect_outliers, run_normality_tests

    # 1. Profil du dataset
    profile = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }

    # 2. Taux de manquants
    missing_info = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        pct_missing = (n_missing / len(df) * 100) if len(df) > 0 else 0.0
        missing_info[col] = {
            "n_missing": n_missing,
            "pct_missing": round(pct_missing, 2),
        }

    # 3. Résumés descriptifs
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    summary_result = summarize_variables(df) if (numeric_cols or categorical_cols) else None
    summary_data = summary_result.model_metrics["tables"]["summary"] if summary_result else []

    # 4. Corrélations (uniquement si au moins 2 variables numériques)
    correlation_result = None
    correlation_plot_base64 = None
    if len(numeric_cols) >= 2:
        cols_to_corr = numeric_cols[:10]
        correlation_result = run_correlation(df, cols_to_corr)
        corr_fig = plot_correlation_heatmap(df, cols_to_corr)
        correlation_plot_base64 = _fig_to_base64(corr_fig)

    # 5. Tests de normalité et outliers
    normality_results = []
    outlier_results = []
    for col in numeric_cols[:5]:
        try:
            norm_result = run_normality_tests(df, col)
            normality_results.append({
                "column": col,
                "is_normal": norm_result.model_metrics["is_normal"],
                "p_values": norm_result.p_values,
            })
        except Exception:  # noqa: BLE001, S110
            pass

        try:
            outlier_result = detect_outliers(df, [col], method="iqr")
            outlier_results.append({
                "column": col,
                "n_outliers": outlier_result.estimates["n_outliers"],
            })
        except Exception:  # noqa: BLE001, S110
            pass

    # 6. Graphiques (histogrammes)
    histogram_plots = {}
    for col in numeric_cols[:5]:
        try:
            fig = plot_histogram(df, col)
            histogram_plots[col] = _fig_to_base64(fig)
        except Exception:  # noqa: BLE001, S110
            pass

    # 7. Boxplots
    boxplot_plots = {}
    if categorical_cols and numeric_cols:
        for cat_col in categorical_cols[:2]:
            for num_col in numeric_cols[:2]:
                try:
                    fig = plot_boxplot(df, x=cat_col, y=num_col)
                    key = f"{num_col}_by_{cat_col}"
                    boxplot_plots[key] = _fig_to_base64(fig)
                except Exception:  # noqa: BLE001, S110
                    pass

    # Préparation des données pour le template
    eda_data = {
        "dataset_name": data_dictionary.dataset_name,
        "profile": profile,
        "missing_info": missing_info,
        "summary_data": summary_data,
        "correlation_result": correlation_result,
        "correlation_plot": correlation_plot_base64,
        "normality_results": normality_results,
        "outlier_results": outlier_results,
        "histogram_plots": histogram_plots,
        "boxplot_plots": boxplot_plots,
    }

    if output_format == "html":
        return _generate_eda_html_report(eda_data)
    else:
        return _generate_eda_markdown_report(eda_data)


def _fig_to_base64(fig: matplotlib.figure.Figure) -> str:
    """Convertit une figure matplotlib en chaîne base64 (format PNG)."""
    import base64
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def _generate_eda_html_report(eda_data: dict) -> str:
    """Génère un rapport EDA HTML via Jinja2."""
    env = Environment(
        loader=FileSystemLoader(_TEMPLATES_DIR),
        autoescape=True,
    )
    template = env.get_template("eda_report.html.j2")
    return template.render(**eda_data)


def _generate_eda_markdown_report(eda_data: dict) -> str:
    """Génère un rapport EDA Markdown."""
    lines = [
        f"# Rapport EDA — {eda_data['dataset_name']}",
        "",
        "## Profil du dataset",
        "",
        f"- **Nombre de lignes :** {eda_data['profile']['n_rows']}",
        f"- **Nombre de colonnes :** {eda_data['profile']['n_columns']}",
        "",
        "### Types de données",
        "",
    ]

    for col, dtype in eda_data["profile"]["dtypes"].items():
        lines.append(f"- **{col}** : {dtype}")
    lines.append("")

    # Manquants
    lines.extend(
        [
            "## Valeurs manquantes",
            "",
            "| Colonne | Nb manquants | % manquants |",
            "|---------|--------------|-------------|",
        ]
    )
    for col, info in eda_data["missing_info"].items():
        lines.append(f"| {col} | {info['n_missing']} | {info['pct_missing']:.2f}% |")
    lines.append("")

    # Résumé descriptif
    if eda_data["summary_data"]:
        lines.extend(
            [
                "## Résumé descriptif",
                "",
            ]
        )
        for summary in eda_data["summary_data"]:
            lines.append(f"### {summary['variable']}")
            lines.append("")
            for key, value in summary.items():
                if key != "variable":
                    lines.append(f"- **{key}** : {value}")
            lines.append("")

    # Corrélations
    if eda_data["correlation_result"]:
        lines.extend(
            [
                "## Corrélations",
                "",
            ]
        )
        if eda_data["correlation_plot"]:
            lines.append(f"![Heatmap de corrélation]({eda_data['correlation_plot']})")
            lines.append("")

    # Normalité
    if eda_data["normality_results"]:
        lines.extend(
            [
                "## Tests de normalité",
                "",
            ]
        )
        for norm in eda_data["normality_results"]:
            status = "✅ Normale" if norm["is_normal"] else "❌ Non normale"
            lines.append(f"- **{norm['column']}** : {status}")
        lines.append("")

    # Outliers
    if eda_data["outlier_results"]:
        lines.extend(
            [
                "## Détection d'outliers (IQR)",
                "",
            ]
        )
        for outlier in eda_data["outlier_results"]:
            lines.append(
                f"- **{outlier['column']}** : {outlier['n_outliers']} outlier(s)"
            )
        lines.append("")

    # Histogrammes
    if eda_data["histogram_plots"]:
        lines.extend(
            [
                "## Distributions",
                "",
            ]
        )
        for col, img_base64 in eda_data["histogram_plots"].items():
            lines.append(f"### {col}")
            lines.append("")
            lines.append(f"![Histogramme de {col}]({img_base64})")
            lines.append("")

    # Boxplots
    if eda_data["boxplot_plots"]:
        lines.extend(
            [
                "## Boxplots",
                "",
            ]
        )
        for key, img_base64 in eda_data["boxplot_plots"].items():
            lines.append(f"### {key}")
            lines.append("")
            lines.append(f"![Boxplot {key}]({img_base64})")
            lines.append("")

    return "\n".join(lines)
