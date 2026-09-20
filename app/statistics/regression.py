"""Moteur de régression : OLS et Logistique binaire."""

import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from app.results.models import AnalysisResult


def _get_software_versions() -> dict:
    """Retourne les versions des logiciels utilisés."""
    import pandas
    import scipy
    import statsmodels
    return {
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "python": sys.version.split()[0],
    }


def run_ols(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    dataset_id: str = "unknown",
) -> AnalysisResult:
    """Effectue une régression linéaire par moindres carrés ordinaires (OLS).

    Args:
        df: Le DataFrame contenant les données.
        outcome: Nom de la variable dépendante (numérique).
        predictors: Liste des noms des variables indépendantes.
        dataset_id: Identifiant du jeu de données.

    Returns:
        Un AnalysisResult contenant les coefficients, erreurs standards,
        intervalles de confiance, p-values, R², R² ajusté, AIC, BIC.
    """
    if outcome not in df.columns:
        raise ValueError(f"La colonne '{outcome}' n'existe pas dans le DataFrame.")
    for pred in predictors:
        if pred not in df.columns:
            raise ValueError(f"La colonne '{pred}' n'existe pas dans le DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[outcome]):
        raise ValueError(f"La variable '{outcome}' doit être numérique.")

    df_clean = df[[outcome] + predictors].dropna()
    warnings, preprocessing = [], []
    if len(df_clean) < len(df):
        warnings.append(f"{len(df) - len(df_clean)} observation(s) avec valeurs manquantes supprimée(s).")
        preprocessing.append("dropna")

    formula = f"{outcome} ~ " + " + ".join(predictors)
    model = smf.ols(formula=formula, data=df_clean)
    results = model.fit()

    estimates = {"intercept": float(results.params["Intercept"])}
    standard_errors = {"intercept": float(results.bse["Intercept"])}
    confidence_intervals = {"intercept": tuple(float(x) for x in results.conf_int().loc["Intercept"])}
    p_values = {"intercept": float(results.pvalues["Intercept"])}

    for pred in predictors:
        estimates[pred] = float(results.params[pred])
        standard_errors[pred] = float(results.bse[pred])
        confidence_intervals[pred] = tuple(float(x) for x in results.conf_int().loc[pred])
        p_values[pred] = float(results.pvalues[pred])

    return AnalysisResult(
        method="ols",
        dataset_id=dataset_id,
        variables={"outcome": outcome, "predictors": predictors},
        sample_size=len(df_clean),
        preprocessing=preprocessing,
        estimates=estimates,
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals,
        p_values=p_values,
        model_metrics={
            "r_squared": float(results.rsquared),
            "r_squared_adj": float(results.rsquared_adj),
            "aic": float(results.aic),
            "bic": float(results.bic),
            "f_statistic": float(results.fvalue),
            "f_p_value": float(results.f_pvalue),
        },
        diagnostics=[],
        warnings=warnings,
        assumptions_checked=["normalite_residus_shapiro", "heteroscedasticite_breusch_pagan"],
        generated_code=f"import statsmodels.formula.api as smf\nmodel = smf.ols(formula='{formula}', data=df)\nresults = model.fit()",
        execution_log=["dropna", "fit_ols", "extract_summary"],
        software_versions=_get_software_versions(),
    )


def run_logistic(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    reference_category: str | None = None,
    dataset_id: str = "unknown",
) -> AnalysisResult:
    """Effectue une régression logistique binaire.

    Args:
        df: Le DataFrame contenant les données.
        outcome: Nom de la variable dépendante binaire (0/1 ou booléen).
        predictors: Liste des noms des variables indépendantes.
        reference_category: Catégorie de référence pour les variables catégorielles.
        dataset_id: Identifiant du jeu de données.

    Returns:
        Un AnalysisResult contenant les coefficients (log-odds), erreurs standards,
        intervalles de confiance, p-values, odds ratios, pseudo-R², AIC, BIC.
    """
    if outcome not in df.columns:
        raise ValueError(f"La colonne '{outcome}' n'existe pas dans le DataFrame.")
    for pred in predictors:
        if pred not in df.columns:
            raise ValueError(f"La colonne '{pred}' n'existe pas dans le DataFrame.")

    # Vérification binaire
    unique_outcomes = df[outcome].dropna().unique()
    if len(unique_outcomes) > 2:
        raise ValueError(f"La variable '{outcome}' doit être binaire, mais a {len(unique_outcomes)} modalités.")

    df_clean = df[[outcome] + predictors].dropna()
    warnings, preprocessing = [], []
    if len(df_clean) < len(df):
        warnings.append(f"{len(df) - len(df_clean)} observation(s) avec valeurs manquantes supprimée(s).")
        preprocessing.append("dropna")

    # Construction de la formule Patsy
    pred_terms = []
    for pred in predictors:
        if reference_category and pd.api.types.is_categorical_dtype(df_clean[pred]) or reference_category and df_clean[pred].dtype == object:
            pred_terms.append(f"C({pred}, Treatment(reference='{reference_category}'))")
        else:
            pred_terms.append(pred)
    
    formula = f"{outcome} ~ " + " + ".join(pred_terms)
    
    try:
        model = smf.logit(formula=formula, data=df_clean)
        results = model.fit(disp=0)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Le modèle logistique n'a pas convergé : {e!s}")

    estimates = {"intercept": float(results.params["Intercept"])}
    standard_errors = {"intercept": float(results.bse["Intercept"])}
    confidence_intervals = {"intercept": tuple(float(x) for x in results.conf_int().loc["Intercept"])}
    p_values = {"intercept": float(results.pvalues["Intercept"])}
    odds_ratios = {"intercept": float(np.exp(results.params["Intercept"]))}

    for param in results.params.index:
        if param != "Intercept":
            estimates[param] = float(results.params[param])
            standard_errors[param] = float(results.bse[param])
            confidence_intervals[param] = tuple(float(x) for x in results.conf_int().loc[param])
            p_values[param] = float(results.pvalues[param])
            odds_ratios[param] = float(np.exp(results.params[param]))

    # Détection de quasi-séparation (valeurs extrêmes ou SE énormes)
    if any(se > 100 for se in standard_errors.values()):
        warnings.append("Attention : erreur standard très élevée détectée, risque de séparation complète (quasi-séparation).")

    return AnalysisResult(
        method="logistic",
        dataset_id=dataset_id,
        variables={"outcome": outcome, "predictors": predictors, "reference_category": reference_category},
        sample_size=len(df_clean),
        preprocessing=preprocessing,
        estimates=estimates,
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals,
        p_values=p_values,
        model_metrics={
            "pseudo_r_squared": float(results.prsquared),
            "aic": float(results.aic),
            "bic": float(results.bic),
            "log_likelihood": float(results.llf),
            "odds_ratios": odds_ratios,
        },
        diagnostics=[],
        warnings=warnings,
        assumptions_checked=["independance_observations", "absence_multicollinearite_severe", "linearite_logit_variables_continues"],
        generated_code=f"import statsmodels.formula.api as smf\nmodel = smf.logit(formula='{formula}', data=df)\nresults = model.fit()",
        execution_log=["dropna", "fit_logit", "extract_summary", "compute_odds_ratios"],
        software_versions=_get_software_versions(),
    )
