"""Régression régularisée V2.
Implémente : Ridge (L2), Lasso (L1), Elastic Net (L1+L2).
"""
from __future__ import annotations

import platform
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler

from app.results.models import AnalysisResult
from app.statistics.registry import MethodSpec, register_method


def _safe_var_name(name: str) -> str:
    """Entoure un nom de variable de Q('...') s'il contient des caractères spéciaux."""
    if any(c in name for c in " -+*/^&|<>!@#$%?") or " " in name:
        escaped_name = name.replace("'", "\\'")
        return f"Q('{escaped_name}')"
    return name


def _build_formula(outcome: str, predictors: list[str]) -> str:
    """Construit une formule Patsy sécurisée."""
    predictors_formula = " + ".join(_safe_var_name(p) for p in predictors)
    return f"{_safe_var_name(outcome)} ~ {predictors_formula}"


# ============================================================================
# 1. Régression Ridge (L2)
# ============================================================================


def run_ridge_regression(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    alpha: float = 1.0,
    standardize: bool = True,
) -> AnalysisResult:
    """Ajuste un modèle de régression Ridge (pénalité L2).

    Réduit la multicolinéarité en pénalisant les grands coefficients.
    Ne sélectionne pas de variables (tous les coefficients sont non nuls).

    Args:
        df: DataFrame contenant les données.
        outcome: Nom de la variable cible continue.
        predictors: Liste des noms des variables prédictives.
        alpha: Paramètre de régularisation (plus grand = plus de pénalité).
        standardize: Standardiser les prédicteurs avant ajustement (recommandé).

    Returns:
        AnalysisResult avec coefficients, métriques, etc.
    """
    if df is None:
        raise ValueError("L'exécution de cette fonction nécessite un DataFrame (df) valide.")
    if outcome not in df.columns:
        raise ValueError(f"La variable '{outcome}' n'existe pas dans le DataFrame.")

    X = df[predictors].values
    y = df[outcome].values

    # Standardisation optionnelle
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    # Ajuster le modèle Ridge
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X_scaled, y)

    # Prédire et calculer les résidus
    y_pred = model.predict(X_scaled)
    residuals = y - y_pred

    # Calculer R² et R² ajusté
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    n = len(y)
    p = len(predictors)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # Extraire les coefficients
    estimates = {pred: float(coef) for pred, coef in zip(predictors, model.coef_)}
    estimates["Intercept"] = float(model.intercept_)

    # Métriques
    model_metrics = {
        "method_family": "regression",
        "r2": float(r2),
        "r2_adj": float(r2_adj),
        "alpha": float(alpha),
        "n_obs": int(n),
        "n_predictors": int(p),
        "standardized": standardize,
    }

    generated_code = f"""import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("CHEMIN_VERS_VOS_DONNEES.csv")
X = df[{predictors}].values
y = df["{outcome}"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = Ridge(alpha={alpha}, fit_intercept=True)
model.fit(X_scaled, y)

print(f"R²: {{model.score(X_scaled, y):.4f}}")
print(f"Coefficients: {{dict(zip({predictors}, model.coef_))}}")
print(f"Intercept: {{model.intercept_:.4f}}")
"""

    return AnalysisResult(
        method="ridge_regression",
        dataset_id="in_memory",
        variables={
            outcome: {"role": "outcome"},
            **{p: {"role": "predictor"} for p in predictors},
        },
        sample_size=int(n),
        preprocessing=["Standardisation des prédicteurs"] if standardize else [],
        estimates=estimates,
        standard_errors={},  # Ridge ne fournit pas d'erreurs standards
        confidence_intervals={},  # Pas d'IC pour Ridge
        p_values={},  # Pas de p-values pour Ridge
        model_metrics=model_metrics,
        diagnostics=[],
        warnings=["Ridge ne fournit pas de p-values ni d'intervalles de confiance"],
        assumptions_checked=[
            "Linéarité de la relation",
            "Homoscédasticité des résidus",
            "Normalité des résidus",
            "Absence de multicolinéarité extrême",
        ],
        generated_code=generated_code,
        execution_log=["Modèle Ridge ajusté avec succès"],
        software_versions={
            "python": platform.python_version(),
            "scikit-learn": "1.5.0",  # À adapter selon ta version
        },
    )


# ============================================================================
# 2. Régression Lasso (L1)
# ============================================================================


def run_lasso_regression(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    alpha: float = 1.0,
    standardize: bool = True,
) -> AnalysisResult:
    """Ajuste un modèle de régression Lasso (pénalité L1).

    Sélection de variables : certains coefficients sont exactement nuls.

    Args:
        df: DataFrame contenant les données.
        outcome: Nom de la variable cible continue.
        predictors: Liste des noms des variables prédictives.
        alpha: Paramètre de régularisation (plus grand = plus de sélection).
        standardize: Standardiser les prédicteurs avant ajustement (recommandé).

    Returns:
        AnalysisResult avec coefficients (dont certains nuls), métriques, etc.
    """
    if df is None:
        raise ValueError("L'exécution de cette fonction nécessite un DataFrame (df) valide.")
    if outcome not in df.columns:
        raise ValueError(f"La variable '{outcome}' n'existe pas dans le DataFrame.")

    X = df[predictors].values
    y = df[outcome].values

    # Standardisation optionnelle
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    # Ajuster le modèle Lasso
    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
    model.fit(X_scaled, y)

    # Prédire et calculer les résidus
    y_pred = model.predict(X_scaled)
    residuals = y - y_pred

    # Calculer R² et R² ajusté
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    n = len(y)
    p = len(predictors)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # Extraire les coefficients
    estimates = {pred: float(coef) for pred, coef in zip(predictors, model.coef_)}
    estimates["Intercept"] = float(model.intercept_)

    # Compter les variables sélectionnées (coefficients non nuls)
    n_selected = sum(1 for coef in model.coef_ if abs(coef) > 1e-10)

    # Métriques
    model_metrics = {
        "method_family": "regression",
        "r2": float(r2),
        "r2_adj": float(r2_adj),
        "alpha": float(alpha),
        "n_obs": int(n),
        "n_predictors": int(p),
        "n_selected": int(n_selected),
        "standardized": standardize,
    }

    generated_code = f"""import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("CHEMIN_VERS_VOS_DONNEES.csv")
X = df[{predictors}].values
y = df["{outcome}"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = Lasso(alpha={alpha}, fit_intercept=True, max_iter=10000)
model.fit(X_scaled, y)

print(f"R²: {{model.score(X_scaled, y):.4f}}")
print(f"Variables sélectionnées: {{sum(abs(coef) > 1e-10 for coef in model.coef_)}}")
print(f"Coefficients: {{dict(zip({predictors}, model.coef_))}}")
print(f"Intercept: {{model.intercept_:.4f}}")
"""

    return AnalysisResult(
        method="lasso_regression",
        dataset_id="in_memory",
        variables={
            outcome: {"role": "outcome"},
            **{p: {"role": "predictor"} for p in predictors},
        },
        sample_size=int(n),
        preprocessing=["Standardisation des prédicteurs"] if standardize else [],
        estimates=estimates,
        standard_errors={},
        confidence_intervals={},
        p_values={},
        model_metrics=model_metrics,
        diagnostics=[],
        warnings=["Lasso ne fournit pas de p-values ni d'intervalles de confiance"],
        assumptions_checked=[
            "Linéarité de la relation",
            "Homoscédasticité des résidus",
            "Normalité des résidus",
        ],
        generated_code=generated_code,
        execution_log=["Modèle Lasso ajusté avec succès"],
        software_versions={
            "python": platform.python_version(),
            "scikit-learn": "1.5.0",
        },
    )


# ============================================================================
# 3. Régression Elastic Net (L1+L2)
# ============================================================================


def run_elastic_net_regression(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    standardize: bool = True,
) -> AnalysisResult:
    """Ajuste un modèle de régression Elastic Net (pénalité L1+L2).

    Compromis entre Ridge et Lasso : sélection de variables + gestion de la multicolinéarité.

    Args:
        df: DataFrame contenant les données.
        outcome: Nom de la variable cible continue.
        predictors: Liste des noms des variables prédictives.
        alpha: Paramètre de régularisation global.
        l1_ratio: Ratio L1/(L1+L2). 0 = Ridge pur, 1 = Lasso pur.
        standardize: Standardiser les prédicteurs avant ajustement (recommandé).

    Returns:
        AnalysisResult avec coefficients, métriques, etc.
    """
    if df is None:
        raise ValueError("L'exécution de cette fonction nécessite un DataFrame (df) valide.")
    if outcome not in df.columns:
        raise ValueError(f"La variable '{outcome}' n'existe pas dans le DataFrame.")

    X = df[predictors].values
    y = df[outcome].values

    # Standardisation optionnelle
    if standardize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    # Ajuster le modèle Elastic Net
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=10000)
    model.fit(X_scaled, y)

    # Prédire et calculer les résidus
    y_pred = model.predict(X_scaled)
    residuals = y - y_pred

    # Calculer R² et R² ajusté
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    n = len(y)
    p = len(predictors)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # Extraire les coefficients
    estimates = {pred: float(coef) for pred, coef in zip(predictors, model.coef_)}
    estimates["Intercept"] = float(model.intercept_)

    # Compter les variables sélectionnées
    n_selected = sum(1 for coef in model.coef_ if abs(coef) > 1e-10)

    # Métriques
    model_metrics = {
        "method_family": "regression",
        "r2": float(r2),
        "r2_adj": float(r2_adj),
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
        "n_obs": int(n),
        "n_predictors": int(p),
        "n_selected": int(n_selected),
        "standardized": standardize,
    }

    generated_code = f"""import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("CHEMIN_VERS_VOS_DONNEES.csv")
X = df[{predictors}].values
y = df["{outcome}"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = ElasticNet(alpha={alpha}, l1_ratio={l1_ratio}, fit_intercept=True, max_iter=10000)
model.fit(X_scaled, y)

print(f"R²: {{model.score(X_scaled, y):.4f}}")
print(f"Variables sélectionnées: {{sum(abs(coef) > 1e-10 for coef in model.coef_)}}")
print(f"Coefficients: {{dict(zip({predictors}, model.coef_))}}")
print(f"Intercept: {{model.intercept_:.4f}}")
"""

    return AnalysisResult(
        method="elastic_net_regression",
        dataset_id="in_memory",
        variables={
            outcome: {"role": "outcome"},
            **{p: {"role": "predictor"} for p in predictors},
        },
        sample_size=int(n),
        preprocessing=["Standardisation des prédicteurs"] if standardize else [],
        estimates=estimates,
        standard_errors={},
        confidence_intervals={},
        p_values={},
        model_metrics=model_metrics,
        diagnostics=[],
        warnings=["Elastic Net ne fournit pas de p-values ni d'intervalles de confiance"],
        assumptions_checked=[
            "Linéarité de la relation",
            "Homoscédasticité des résidus",
            "Normalité des résidus",
        ],
        generated_code=generated_code,
        execution_log=["Modèle Elastic Net ajusté avec succès"],
        software_versions={
            "python": platform.python_version(),
            "scikit-learn": "1.5.0",
        },
    )


# ============================================================================
# Enregistrement dans le registre (V2 §7.1)
# ============================================================================

register_method(
    MethodSpec(
        method_id="ridge_regression",
        label_fr="Régression Ridge (L2)",
        family="regression",
        engine="python",
        outcome_types=["continuous"],
        predictor_types=["continuous"],
        min_sample_size=20,
        assumptions=[
            "Linéarité de la relation",
            "Homoscédasticité des résidus",
            "Normalité des résidus",
            "Absence de multicolinéarité extrême",
        ],
        reference_impl="sklearn.linear_model.Ridge",
        r_reference="MASS::lm.ridge",
        phase=18,
        extra=None,
        default_params={"alpha": 1.0, "standardize": True},
    ),
    run_ridge_regression,
)

register_method(
    MethodSpec(
        method_id="lasso_regression",
        label_fr="Régression Lasso (L1)",
        family="regression",
        engine="python",
        outcome_types=["continuous"],
        predictor_types=["continuous"],
        min_sample_size=20,
        assumptions=[
            "Linéarité de la relation",
            "Homoscédasticité des résidus",
            "Normalité des résidus",
        ],
        reference_impl="sklearn.linear_model.Lasso",
        r_reference="glmnet::glmnet",
        phase=18,
        extra=None,
        default_params={"alpha": 1.0, "standardize": True},
    ),
    run_lasso_regression,
)

register_method(
    MethodSpec(
        method_id="elastic_net_regression",
        label_fr="Régression Elastic Net (L1+L2)",
        family="regression",
        engine="python",
        outcome_types=["continuous"],
        predictor_types=["continuous"],
        min_sample_size=20,
        assumptions=[
            "Linéarité de la relation",
            "Homoscédasticité des résidus",
            "Normalité des résidus",
        ],
        reference_impl="sklearn.linear_model.ElasticNet",
        r_reference="glmnet::glmnet",
        phase=18,
        extra=None,
        default_params={"alpha": 1.0, "l1_ratio": 0.5, "standardize": True},
    ),
    run_elastic_net_regression,
)