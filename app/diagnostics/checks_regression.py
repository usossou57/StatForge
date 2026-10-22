"""Diagnostics de régression V2.
Implémente les vérifications d'hypothèses pour les modèles de régression.
"""
from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any, Literal

import scipy
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from app.results.models import AnalysisResult


@dataclass
class DiagnosticReport:
    """Rapport de diagnostics pour un modèle de régression."""
    
    method: str
    dataset_id: str
    
    # Tests statistiques
    normality_tests: dict[str, dict[str, float]] = field(default_factory=dict)
    homoscedasticity_tests: dict[str, dict[str, float]] = field(default_factory=dict)
    autocorrelation_tests: dict[str, dict[str, float]] = field(default_factory=dict)
    
    # Métriques
    vif: dict[str, float] = field(default_factory=dict)
    cooks_distance: dict[int, float] = field(default_factory=dict)
    leverage: dict[int, float] = field(default_factory=dict)
    
    # Résidus
    residuals: list[float] = field(default_factory=list)
    standardized_residuals: list[float] = field(default_factory=list)
    studentized_residuals: list[float] = field(default_factory=list)
    fitted_values: list[float] = field(default_factory=list)
    
    # Visualisations (chemins des fichiers PNG)
    plots: dict[str, str] = field(default_factory=dict)
    
    # Interprétation
    warnings: list[str] = field(default_factory=list)
    assumptions_met: dict[str, bool] = field(default_factory=dict)
    
    # Code reproductible
    generated_code: str = ""
    
    # Versions
    software_versions: dict[str, str] = field(default_factory=dict)


def run_regression_diagnostics(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    model_type: Literal["ols", "glm", "robust"] = "ols",
    output_dir: str = "diagnostics_output",
) -> DiagnosticReport:
    """Exécute une batterie complète de diagnostics pour un modèle de régression.
    
    Args:
        df: DataFrame contenant les données.
        outcome: Nom de la variable cible.
        predictors: Liste des noms des variables prédictives.
        model_type: Type de modèle ("ols", "glm", "robust").
        output_dir: Répertoire pour sauvegarder les graphiques.
    
    Returns:
        DiagnosticReport avec tous les résultats.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Construire la formule et ajuster le modèle
    formula = f"{outcome} ~ {' + '.join(predictors)}"
    
    if model_type == "ols":
        model = sm.OLS.from_formula(formula, data=df)
        result = model.fit()
    elif model_type == "glm":
        model = sm.GLM.from_formula(formula, data=df, family=sm.families.Gaussian())
        result = model.fit()
    elif model_type == "robust":
        from statsmodels.robust.robust_linear_model import RLM
        model = RLM.from_formula(formula, data=df)
        result = model.fit()
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")
    
    # Extraire les résidus et valeurs ajustées
    residuals = result.resid
    fitted_values = result.fittedvalues
    
    # Résidus standardisés
    standardized_residuals = (residuals - residuals.mean()) / residuals.std()
    
    # Résidus studentisés (externally studentized)
    try:
        studentized_residuals = result.get_influence().resid_studentized_external
    except Exception:
        studentized_residuals = standardized_residuals
    
    # ========================================================================
    # 1. Tests de normalité
    # ========================================================================
    normality_tests = {}
    
    # Shapiro-Wilk (valide pour n < 5000)
    if len(residuals) < 5000:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        normality_tests["shapiro_wilk"] = {
            "statistic": float(shapiro_stat),
            "p_value": float(shapiro_p),
            "normal": bool(shapiro_p > 0.05),
        }
    
    # Jarque-Bera
    jb_result = stats.jarque_bera(residuals)
    jb_stat = float(jb_result[0]) if hasattr(jb_result, '__getitem__') else float(jb_result.statistic)
    jb_p = float(jb_result[1]) if hasattr(jb_result, '__getitem__') else float(jb_result.pvalue)
    normality_tests["jarque_bera"] = {
        "statistic": jb_stat,
        "p_value": jb_p,
        "normal": bool(jb_p > 0.05),
    }
    
    # ========================================================================
    # 2. Tests d'homoscédasticité
    # ========================================================================
    homoscedasticity_tests = {}
    
    # Breusch-Pagan
    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(residuals, result.model.exog)
        homoscedasticity_tests["breusch_pagan"] = {
            "statistic": float(bp_stat),
            "p_value": float(bp_p),
            "homoscedastic": bool(bp_p > 0.05),
        }
    except Exception as e:
        homoscedasticity_tests["breusch_pagan"] = {"error": str(e)}
    
    # White test
    try:
        white_stat, white_p, _, _ = het_white(residuals, result.model.exog)
        homoscedasticity_tests["white"] = {
            "statistic": float(white_stat),
            "p_value": float(white_p),
            "homoscedastic": bool(white_p > 0.05),
        }
    except Exception as e:
        homoscedasticity_tests["white"] = {"error": str(e)}
    
    # ========================================================================
    # 3. Test d'autocorrélation
    # ========================================================================
    autocorrelation_tests = {}
    
    # Durbin-Watson
    dw_stat = durbin_watson(residuals)
    autocorrelation_tests["durbin_watson"] = {
        "statistic": float(dw_stat),
        "no_autocorrelation": bool(1.5 < dw_stat < 2.5),  # Règle empirique
    }
    
    # ========================================================================
    # 4. Multicolinéarité (VIF)
    # ========================================================================
    vif = {}
    if len(predictors) > 1:
        try:
            X = result.model.exog[:, 1:]  # Exclure l'intercept
            for i, pred in enumerate(predictors):
                vif_val = variance_inflation_factor(X, i)
                vif[pred] = float(vif_val)
        except Exception as e:
            vif["error"] = str(e)
    
    # ========================================================================
    # 5. Influence (Cook's distance et leverage)
    # ========================================================================
    cooks_distance = {}
    leverage = {}
    
    try:
        influence = result.get_influence()
        cooks_d, _ = influence.cooks_distance
        leverage_vals = influence.hat_matrix_diag
        
        for i in range(len(cooks_d)):
            cooks_distance[i] = float(cooks_d[i])
            leverage[i] = float(leverage_vals[i])
    except Exception as e:
        cooks_distance["error"] = str(e)
        leverage["error"] = str(e)
    
    # ========================================================================
    # 6. Visualisations
    # ========================================================================
    plots = {}
    
    try:
        # Residuals vs Fitted
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(fitted_values, residuals, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Fitted')
        plot_path = os.path.join(output_dir, "residuals_vs_fitted.png")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        plots["residuals_vs_fitted"] = plot_path
        
        # QQ-plot
        fig, ax = plt.subplots(figsize=(8, 6))
        stats.probplot(standardized_residuals, dist="norm", plot=ax)
        ax.set_title('Normal Q-Q')
        plot_path = os.path.join(output_dir, "qq_plot.png")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        plots["qq_plot"] = plot_path
        
        # Scale-Location (Spread-Location)
        fig, ax = plt.subplots(figsize=(8, 6))
        sqrt_abs_std_resid = np.sqrt(np.abs(standardized_residuals))
        ax.scatter(fitted_values, sqrt_abs_std_resid, alpha=0.6)
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('√|Standardized Residuals|')
        ax.set_title('Scale-Location')
        plot_path = os.path.join(output_dir, "scale_location.png")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        plots["scale_location"] = plot_path
        
        # Residuals vs Leverage
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(list(leverage.values()), standardized_residuals, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized Residuals')
        ax.set_title('Residuals vs Leverage')
        plot_path = os.path.join(output_dir, "residuals_vs_leverage.png")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        plots["residuals_vs_leverage"] = plot_path
        
    except Exception as e:
        plots["error"] = str(e)
    
    # ========================================================================
    # 7. Interprétation et warnings
    # ========================================================================
    warnings = []
    assumptions_met = {}
    
    # Normalité
    if "shapiro_wilk" in normality_tests:
        assumptions_met["normality"] = normality_tests["shapiro_wilk"]["normal"]
        if not assumptions_met["normality"]:
            warnings.append("Les résidus ne suivent pas une distribution normale (Shapiro-Wilk p < 0.05)")
    
    # Homoscédasticité
    if "breusch_pagan" in homoscedasticity_tests and "p_value" in homoscedasticity_tests["breusch_pagan"]:
        assumptions_met["homoscedasticity"] = homoscedasticity_tests["breusch_pagan"]["homoscedastic"]
        if not assumptions_met["homoscedasticity"]:
            warnings.append("Hétéroscédasticité détectée (Breusch-Pagan p < 0.05)")
    
    # Autocorrélation
    assumptions_met["no_autocorrelation"] = autocorrelation_tests["durbin_watson"]["no_autocorrelation"]
    if not assumptions_met["no_autocorrelation"]:
        warnings.append(f"Autocorrélation potentielle détectée (Durbin-Watson = {dw_stat:.3f})")
    
    # Multicolinéarité
    if vif and "error" not in vif:
        max_vif = max(vif.values())
        assumptions_met["no_multicollinearity"] = max_vif < 5.0
        if not assumptions_met["no_multicollinearity"]:
            warnings.append(f"Multicolinéarité élevée détectée (VIF max = {max_vif:.2f})")
    
    # Points influents
    if cooks_distance and "error" not in cooks_distance:
        influential_points = [i for i, d in cooks_distance.items() if d > 4 / len(residuals)]
        assumptions_met["no_influential_points"] = len(influential_points) == 0
        if not assumptions_met["no_influential_points"]:
            warnings.append(f"{len(influential_points)} points influents détectés (Cook's distance > 4/n)")
    
    # ========================================================================
    # 8. Code reproductible
    # ========================================================================
    generated_code = f"""import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
import matplotlib.pyplot as plt

df = pd.read_csv("CHEMIN_VERS_VOS_DONNEES.csv")
formula = "{formula}"

# Ajuster le modèle
model = sm.OLS.from_formula(formula, data=df).fit()

# Résidus
residuals = model.resid
fitted = model.fittedvalues

# Tests de normalité
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"Shapiro-Wilk: stat={{shapiro_stat:.4f}}, p={{shapiro_p:.4f}}")

# Test d'homoscédasticité
bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
print(f"Breusch-Pagan: stat={{bp_stat:.4f}}, p={{bp_p:.4f}}")

# Test d'autocorrélation
dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson: {{dw_stat:.4f}}")

# VIF
X = model.model.exog[:, 1:]
for i, pred in enumerate({predictors}):
    vif = variance_inflation_factor(X, i)
    print(f"VIF {{pred}}: {{vif:.2f}}")

# Cook's distance
influence = model.get_influence()
cooks_d, _ = influence.cooks_distance
print(f"Points influents (Cook's > 4/n): {{sum(cooks_d > 4/len(residuals))}}")

# Visualisations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].scatter(fitted, residuals, alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_title('Residuals vs Fitted')

stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q')

sqrt_abs_resid = np.sqrt(np.abs(residuals))
axes[1, 0].scatter(fitted, sqrt_abs_resid, alpha=0.6)
axes[1, 0].set_title('Scale-Location')

leverage = influence.hat_matrix_diag
axes[1, 1].scatter(leverage, residuals, alpha=0.6)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_title('Residuals vs Leverage')

plt.tight_layout()
plt.savefig("diagnostics.png", dpi=150)
plt.show()
"""
    
    return DiagnosticReport(
        method=f"{model_type}_regression_diagnostics",
        dataset_id="in_memory",
        normality_tests=normality_tests,
        homoscedasticity_tests=homoscedasticity_tests,
        autocorrelation_tests=autocorrelation_tests,
        vif=vif,
        cooks_distance=cooks_distance,
        leverage=leverage,
        residuals=residuals.tolist(),
        standardized_residuals=standardized_residuals.tolist(),
        studentized_residuals=studentized_residuals.tolist() if hasattr(studentized_residuals, '__iter__') else [],
        fitted_values=fitted_values.tolist(),
        plots=plots,
        warnings=warnings,
        assumptions_met=assumptions_met,
        generated_code=generated_code,
        software_versions={
            "python": platform.python_version(),
            "statsmodels": sm.__version__,
            "scipy": scipy.__version__,
        },
    )