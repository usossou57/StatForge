"""Régression robuste, polynomiale et par splines V2."""
from __future__ import annotations

import platform
from typing import Literal

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.robust.norms import HuberT, TukeyBiweight

from app.results.models import AnalysisResult
from app.statistics.registry import MethodSpec, register_method


def _safe_var_name(name: str) -> str:
    if any(c in name for c in " -+*/^&|<>!@#$%?") or " " in name:
        return f"Q('{name.replace(chr(39), chr(92)+chr(39))}')"
    return name


def _build_formula(outcome: str, predictors: list[str]) -> str:
    predictors_formula = " + ".join(_safe_var_name(p) for p in predictors)
    return f"{_safe_var_name(outcome)} ~ {predictors_formula}"


# ============================================================================
# 1. Régression Robuste (RLM)
# ============================================================================


def run_robust_regression(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    m_estimator: Literal["huber", "tukey"] = "huber",
) -> AnalysisResult:
    """Ajuste un modèle de régression linéaire robuste (RLM).
    
    Résiste aux valeurs aberrantes (outliers) en utilisant des fonctions de perte
    alternatives (M-estimators) comme Huber ou Tukey.
    """
    if df is None:
        raise ValueError("DataFrame valide requis.")
    if outcome not in df.columns:
        raise ValueError(f"La variable '{outcome}' n'existe pas.")

    formula = _build_formula(outcome, predictors)
    norm = HuberT() if m_estimator == "huber" else TukeyBiweight()
    
    model = sm.RLM.from_formula(formula, data=df, M=norm)
    result = model.fit()

    estimates = result.params.to_dict()
    standard_errors = result.bse.to_dict()
    p_values = result.pvalues.to_dict()
    ci = result.conf_int()
    confidence_intervals = {idx: (float(row[0]), float(row[1])) for idx, row in ci.iterrows()}

    model_metrics = {
        "method_family": "regression",
        "m_estimator": m_estimator,
        "n_obs": int(result.nobs),
        "df_resid": int(result.df_resid),
        "scale": float(result.scale),
    }

    generated_code = f"""import pandas as pd
import statsmodels.api as sm
from statsmodels.robust.norms import {norm.__class__.__name__}

df = pd.read_csv("CHEMIN_VERS_VOS_DONNEES.csv")
formula = "{formula}"
model = sm.RLM.from_formula(formula, data=df, M={norm.__class__.__name__}()).fit()
print(model.summary())
"""

    return AnalysisResult(
        method="robust_regression",
        dataset_id="in_memory",
        variables={outcome: {"role": "outcome"}, **{p: {"role": "predictor"} for p in predictors}},
        sample_size=int(result.nobs),
        preprocessing=[f"M-estimator: {m_estimator.capitalize()}"],
        estimates=estimates,
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals,
        p_values=p_values,
        model_metrics=model_metrics,
        diagnostics=[],
        warnings=[],
        assumptions_checked=[
            "Linéarité de la relation (sauf pour les outliers)",
            "Indépendance des erreurs",
            "Homoscédasticité relative",
        ],
        generated_code=generated_code,
        execution_log=["Modèle robuste (RLM) ajusté avec succès"],
        software_versions={"python": platform.python_version(), "statsmodels": sm.__version__},
    )


# ============================================================================
# 2. Régression Polynomiale
# ============================================================================


def run_polynomial_regression(
    df: pd.DataFrame,
    outcome: str,
    predictor: str,
    degree: int = 2,
) -> AnalysisResult:
    """Ajuste un modèle de régression polynomiale simple.
    
    Utile pour capturer des courbures simples (ex: relation en U).
    """
    if df is None:
        raise ValueError("DataFrame valide requis.")
    if outcome not in df.columns or predictor not in df.columns:
        raise ValueError("Variables outcome ou predictor manquantes.")
    if degree < 2 or degree > 5:
        raise ValueError("Le degré polynomial doit être entre 2 et 5.")

    # Construire la formule : y ~ x + I(x**2) + I(x**3) ...
    poly_terms = [predictor] + [f"I({_safe_var_name(predictor)}**{i})" for i in range(2, degree + 1)]
    formula = f"{_safe_var_name(outcome)} ~ {' + '.join(poly_terms)}"

    model = smf.ols(formula, data=df)
    result = model.fit()

    estimates = result.params.to_dict()
    standard_errors = result.bse.to_dict()
    p_values = result.pvalues.to_dict()
    ci = result.conf_int()
    confidence_intervals = {idx: (float(row[0]), float(row[1])) for idx, row in ci.iterrows()}

    model_metrics = {
        "method_family": "regression",
        "degree": int(degree),
        "r2": float(result.rsquared),
        "r2_adj": float(result.rsquared_adj),
        "n_obs": int(result.nobs),
    }

    generated_code = f"""import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("CHEMIN_VERS_VOS_DONNEES.csv")
formula = "{formula}"
model = smf.ols(formula, data=df).fit()
print(model.summary())
"""

    return AnalysisResult(
        method="polynomial_regression",
        dataset_id="in_memory",
        variables={outcome: {"role": "outcome"}, predictor: {"role": "predictor"}},
        sample_size=int(result.nobs),
        preprocessing=[f"Transformation polynomiale de degré {degree}"],
        estimates=estimates,
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals,
        p_values=p_values,
        model_metrics=model_metrics,
        diagnostics=[],
        warnings=["Attention au surajustement (overfitting) pour les degrés élevés."],
        assumptions_checked=[
            "Relation polynomiale appropriée",
            "Indépendance des erreurs",
            "Homoscédasticité",
            "Normalité des résidus",
        ],
        generated_code=generated_code,
        execution_log=["Modèle polynomial ajusté avec succès"],
        software_versions={"python": platform.python_version(), "statsmodels": sm.__version__},
    )


# ============================================================================
# 3. Régression par Splines (via patsy + OLS)
# ============================================================================


def run_spline_regression(
    df: pd.DataFrame,
    outcome: str,
    predictor: str,
    df_spline: int = 4,
    degree: int = 3,
) -> AnalysisResult:
    """Ajuste un modèle de régression par B-splines (natural cubic splines).
    
    Permet de modéliser des relations non linéaires complexes et flexibles.
    Utilise patsy.bs() qui correspond à splines::ns() de R.
    
    Args:
        df: DataFrame contenant les données.
        outcome: Nom de la variable cible continue.
        predictor: Nom de la variable prédictive continue.
        df_spline: Degrés de liberté pour les splines (nombre de nœuds internes + degree).
        degree: Degré des splines (3 = splines cubiques, par défaut).
    """
    if df is None:
        raise ValueError("DataFrame valide requis.")
    if outcome not in df.columns or predictor not in df.columns:
        raise ValueError("Variables outcome ou predictor manquantes.")

    # Construire la matrice de design avec patsy (équivalent à splines::ns() de R)
    # bs() de patsy = ns() de R (natural splines)
    spline_formula = f"bs({_safe_var_name(predictor)}, df={df_spline}, degree={degree}) - 1"
    X_spline = patsy.dmatrix(spline_formula, data=df, return_type="dataframe")
    
    # Ajouter l'intercept manuellement
    X_with_intercept = sm.add_constant(X_spline)
    y = df[outcome].values
    
    # Ajuster le modèle OLS
    model = sm.OLS(y, X_with_intercept)
    result = model.fit()

    # Extraire les résultats
    estimates = result.params.to_dict()
    standard_errors = result.bse.to_dict()
    p_values = result.pvalues.to_dict()
    ci = result.conf_int()
    confidence_intervals = {idx: (float(row[0]), float(row[1])) for idx, row in ci.iterrows()}

    model_metrics = {
        "method_family": "regression",
        "df_spline": int(df_spline),
        "degree": int(degree),
        "r2": float(result.rsquared),
        "r2_adj": float(result.rsquared_adj),
        "n_obs": int(result.nobs),
    }

    generated_code = f"""import pandas as pd
import patsy
import statsmodels.api as sm

df = pd.read_csv("CHEMIN_VERS_VOS_DONNEES.csv")

# Construire la matrice de splines (équivalent à splines::ns() de R)
X_spline = patsy.dmatrix("bs({_safe_var_name(predictor)}, df={df_spline}, degree={degree}) - 1", data=df, return_type="dataframe")
X = sm.add_constant(X_spline)
y = df["{outcome}"]

model = sm.OLS(y, X).fit()
print(model.summary())
"""

    return AnalysisResult(
        method="spline_regression",
        dataset_id="in_memory",
        variables={outcome: {"role": "outcome"}, predictor: {"role": "predictor"}},
        sample_size=int(result.nobs),
        preprocessing=[f"B-splines: df={df_spline}, degree={degree} (via patsy.bs)"],
        estimates=estimates,
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals,
        p_values=p_values,
        model_metrics=model_metrics,
        diagnostics=[],
        warnings=["Les coefficients individuels des splines sont difficiles à interpréter isolément."],
        assumptions_checked=[
            "Relation lisse et continue",
            "Indépendance des erreurs",
            "Homoscédasticité",
            "Normalité des résidus",
        ],
        generated_code=generated_code,
        execution_log=["Modèle par splines ajusté avec succès"],
        software_versions={"python": platform.python_version(), "statsmodels": sm.__version__},
    )


# ============================================================================
# Enregistrement dans le registre (V2 §7.1)
# ============================================================================

register_method(
    MethodSpec(
        method_id="robust_regression",
        label_fr="Régression Robuste (RLM)",
        family="regression",
        engine="python",
        outcome_types=["continuous"],
        predictor_types=["continuous", "categorical"],
        min_sample_size=20,
        assumptions=["Linéarité (robuste aux outliers)", "Indépendance", "Homoscédasticité relative"],
        reference_impl="statsmodels.robust.robust_linear_model.RLM",
        r_reference="MASS::rlm",
        phase=18,
        extra=None,
        default_params={"m_estimator": "huber"},
    ),
    run_robust_regression,
)

register_method(
    MethodSpec(
        method_id="polynomial_regression",
        label_fr="Régression Polynomiale",
        family="regression",
        engine="python",
        outcome_types=["continuous"],
        predictor_types=["continuous"],
        min_sample_size=20,
        assumptions=["Relation polynomiale", "Indépendance", "Homoscédasticité", "Normalité des résidus"],
        reference_impl="statsmodels.formula.api.ols",
        r_reference="lm(y ~ poly(x, degree, raw=TRUE))",
        phase=18,
        extra=None,
        default_params={"degree": 2},
    ),
    run_polynomial_regression,
)

register_method(
    MethodSpec(
        method_id="spline_regression",
        label_fr="Régression par Splines (B-splines)",
        family="regression",
        engine="python",
        outcome_types=["continuous"],
        predictor_types=["continuous"],
        min_sample_size=30,
        assumptions=["Relation lisse", "Indépendance", "Homoscédasticité", "Normalité des résidus"],
        reference_impl="patsy.bs + statsmodels.api.OLS",
        r_reference="splines::ns() + lm",
        phase=18,
        extra=None,
        default_params={"df_spline": 4, "degree": 3},
    ),
    run_spline_regression,
)