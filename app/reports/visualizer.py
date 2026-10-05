"""Génération de visualisations statistiques adaptées à chaque méthode.

Ce module produit des graphiques pertinents selon le type d'analyse :
- Régressions (linear, logistic, firth, multinomial) : forest plot des coefficients
- Chi-carré : résumé visuel du test
- t-test / ANOVA : résumé visuel des statistiques par groupe
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.results.models import AnalysisResult


# ──────────────────────────────────────────────────────────────────────
# Fonction principale : dispatch selon la méthode
# ──────────────────────────────────────────────────────────────────────

def generate_coefficient_plot(
    result: AnalysisResult,
    output_dir: str = "reports/plots",
    filename: str = "coef_plot.png",
) -> str:
    """Génère un graphique adapté à la méthode d'analyse."""
    method = result.method.lower()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename

    if "chi_square" in method or "chi2" in method:
        fig = _generate_chi_square_plot(result)
    elif method in ("t_test", "anova"):
        fig = _generate_group_comparison_plot(result)
    else:
        fig = _generate_forest_plot(result)

    if fig is None:
        raise ValueError(
            f"Aucun graphique disponible pour la méthode '{result.method}'."
        )

    fig.savefig(file_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return str(file_path)


# ──────────────────────────────────────────────────────────────────────
# Forest plot (régressions)
# ──────────────────────────────────────────────────────────────────────

def _generate_forest_plot(result: AnalysisResult) -> plt.Figure:
    """Forest plot des coefficients avec intervalles de confiance."""
    coefficients = []
    for var_name in result.estimates.keys():
        estimate = result.estimates[var_name]
        ci = result.confidence_intervals.get(var_name, (None, None))
        coefficients.append({
            "variable": var_name,
            "estimate": estimate,
            "ci_lower": ci[0] if ci[0] is not None else estimate,
            "ci_upper": ci[1] if ci[1] is not None else estimate,
        })

    df_coef = pd.DataFrame(coefficients)
    if df_coef.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, max(6, len(df_coef) * 0.6)))
    y_pos = range(len(df_coef))

    ax.errorbar(
        df_coef["estimate"], y_pos,
        xerr=[
            df_coef["estimate"] - df_coef["ci_lower"],
            df_coef["ci_upper"] - df_coef["estimate"],
        ],
        fmt="o", color="black", ecolor="gray", capsize=5, markersize=8, linewidth=2,
    )
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_coef["variable"])
    ax.set_xlabel("Estimation", fontsize=12)
    
    method_title = result.method.replace("_", " ").title()
    ax.set_title(
        f"Coefficients estimés — {method_title}",
        fontsize=14, fontweight="bold",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────
# Heatmap de contingence (chi-carré)
# ──────────────────────────────────────────────────────────────────────

def _generate_chi_square_plot(result: AnalysisResult) -> plt.Figure:
    """Résumé visuel du test du chi-carré."""
    fig, ax = plt.subplots(figsize=(8, 6))

    var_names = list(result.variables.keys())
    if len(var_names) < 2:
        ax.text(
            0.5, 0.5,
            "Variables insuffisantes pour afficher un tableau croisé.",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig

    var1_name = result.variables[var_names[0]].get("name", var_names[0])
    var2_name = result.variables[var_names[1]].get("name", var_names[1])

    chi2 = result.estimates.get("chi2", 0)
    dof = result.estimates.get("dof", 1)
    p_value = result.p_values.get("chi2", 1.0)

    # Correction : extraire le backslash dans une variable
    sig_symbol = "✅" if p_value < 0.05 else "❌"
    sig_text = "Association significative" if p_value < 0.05 else "Pas d'association significative"
    
    summary_lines = [
        "Test du chi-carré d'indépendance",
        "",
        f"Variable 1 : {var1_name}",
        f"Variable 2 : {var2_name}",
        "",
        f"χ² = {chi2:.4f}",
        f"ddl = {dof}",
        f"p-value = {p_value:.4e}",
        "",
        f"{sig_symbol} {sig_text}",
    ]
    summary_text = "\n".join(summary_lines)

    ax.set_axis_off()
    ax.text(
        0.5, 0.5, summary_text,
        ha="center", va="center", transform=ax.transAxes,
        fontsize=14, family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    
    method_title = result.method.replace("_", " ").title()
    ax.set_title(
        f"Résultat du test — {method_title}",
        fontsize=14, fontweight="bold",
        pad=20,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
# Boxplot par groupe (t-test / ANOVA)
# ──────────────────────────────────────────────────────────────────────

def _generate_group_comparison_plot(result: AnalysisResult) -> plt.Figure:
    """Résumé visuel pour t-test ou ANOVA."""
    fig, ax = plt.subplots(figsize=(10, 6))

    var_names = list(result.variables.keys())
    outcome_name = None
    group_name = None
    for v in var_names:
        info = result.variables[v]
        if info.get("role") == "outcome":
            outcome_name = info.get("name", v)
        elif info.get("role") == "predictor":
            group_name = info.get("name", v)

    if outcome_name is None or group_name is None:
        ax.text(
            0.5, 0.5,
            "Variables outcome/group non trouvées.",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig

    mean_a = result.model_metrics.get("mean_group_a", result.estimates.get("mean_a", 0))
    mean_b = result.model_metrics.get("mean_group_b", result.estimates.get("mean_b", 0))
    n_a = result.model_metrics.get("n_group_a", 0)
    n_b = result.model_metrics.get("n_group_b", 0)
    n_groups = result.model_metrics.get("n_groups", 2)

    # Construction des lignes du résumé (SANS backslash dans les f-strings)
    lines = [
        f"Comparaison de moyennes — {result.method.upper()}",
        "",
        f"Outcome : {outcome_name}",
        f"Groupe : {group_name}",
        "",
    ]

    if n_groups == 2:
        lines.extend([
            f"Groupe A : moyenne = {mean_a:.4f} (n={n_a})",
            f"Groupe B : moyenne = {mean_b:.4f} (n={n_b})",
            "",
        ])
    else:
        lines.append(f"Nombre de groupes : {n_groups}")
        lines.append("")

    # Statistiques du test
    if "t_statistic" in result.estimates:
        t_stat = result.estimates["t_statistic"]
        p_value = result.p_values.get("t_test", 1.0)
        sig_symbol = "✅" if p_value < 0.05 else "❌"
        sig_text = "Différence significative" if p_value < 0.05 else "Pas de différence significative"
        lines.extend([
            f"t = {t_stat:.4f}",
            f"p-value = {p_value:.4e}",
            "",
            f"{sig_symbol} {sig_text}",
        ])
    elif "f_statistic" in result.estimates:
        f_stat = result.estimates["f_statistic"]
        p_value = result.p_values.get("anova", 1.0)
        sig_symbol = "✅" if p_value < 0.05 else "❌"
        sig_text = "Différence significative" if p_value < 0.05 else "Pas de différence significative"
        lines.extend([
            f"F = {f_stat:.4f}",
            f"p-value = {p_value:.4e}",
            "",
            f"{sig_symbol} {sig_text}",
        ])

    summary_text = "\n".join(lines)

    ax.set_axis_off()
    ax.text(
        0.5, 0.5, summary_text,
        ha="center", va="center", transform=ax.transAxes,
        fontsize=13, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )
    
    method_title = result.method.replace("_", " ").title()
    ax.set_title(
        f"Résultat — {method_title}",
        fontsize=14, fontweight="bold",
        pad=20,
    )
    return fig
