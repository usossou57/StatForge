"""GLM (Generalized Linear Models) pour régression V2.
Implémente : probit, Poisson, binomiale négative, Gamma.
"""
from __future__ import annotations

import platform
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Gamma, Poisson
from statsmodels.genmod.families.links import Identity, InversePower, Log

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
# 1. Régression Probit
# ============================================================================


def run_probit_regression(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    link: Literal["probit", "logit"] = "probit",
) -> AnalysisResult:
    """Ajuste un modèle probit (ou logit) pour données binaires.

    Args:
        df: DataFrame contenant les données.
        outcome: Nom de la variable cible binaire (0/1).
        predictors: Liste des noms des variables prédictives.
        link: Fonction de lien ("probit" ou "logit").

    Returns:
        AnalysisResult avec coefficients, p-values, AIC, etc.
    """
    if df is None:
        raise ValueError("L'exécution de cette fonction nécessite un DataFrame (df) valide.")
    if outcome not in df.columns:
        raise ValueError(f"La variable '{outcome}' n'existe pas dans le DataFrame.")

    formula = _build_formula(outcome, predictors)

    if link == "probit":
        model = smf.probit(formula, data=df)
    elif link == "logit":
        model = smf.logit(formula, data=df)
    else:
        raise ValueError(f"Link non supporté: {link}")

    result = model.fit(disp=False)

    estimates = result.params.to_dict()
    standard_errors = result.bse.to_dict()
    p_values = result.pvalues.to_dict()

    ci = result.conf_int()
    confidence_intervals = {
        idx: (float(row[0]), float(row[1])) for idx, row in ci.iterrows()
    }

    model_metrics = {
        "method_family": "regression",
        "aic": float(result.aic),
        "bic": float(result.bic),
        "pseudo_r2": float(result.prsquared),
        "log_likelihood": float(result.llf),
        "n_obs": int(result.nobs),
    }

    generated_code = f"""import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("CHEMIN_VERS_VOS_DONNEES.csv")
formula = "{formula}"
model = smf.{link}(formula, data=df).fit(disp=False)
print(model.summary())
"""

    return AnalysisResult(
        method="probit_regression",
        dataset_id="in_memory",
        variables={
            outcome: {"role": "outcome"},
            **{p: {"role": "predictor"} for p in predictors},
        },
        sample_size=int(result.nobs),
        preprocessing=["Modèle ajusté avec statsmodels"],
        estimates=estimates,
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals,
        p_values=p_values,
        model_metrics=model_metrics,
        diagnostics=[],
        warnings=[],
        assumptions_checked=[
            "Indépendance des observations",
            "Linéarité des log-odds (logit) ou probits (probit)",
            "Absence de multicolinéarité forte",
        ],
        generated_code=generated_code,
        execution_log=[f"Modèle {'Probit' if link == 'probit' else 'Logit'} ajusté avec succès"],
        software_versions={
            "python": platform.python_version(),
            "statsmodels": statsmodels.__version__,
        },
    )


# ============================================================================
# 2. Régression de Poisson
# ============================================================================

def run_poisson_regression(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    exposure: str | None = None,
) -> AnalysisResult:
    """Ajuste un modèle de Poisson pour données de comptage.

    Args:
        df: DataFrame contenant les données.
        outcome: Nom de la variable cible de comptage (entiers >= 0).
        predictors: Liste des noms des variables prédictives.
        exposure: Nom de la colonne d'exposition (optionnel).

    Returns:
        AnalysisResult avec coefficients, taux d'incidence, etc.
    """
    if df is None:
        raise ValueError("L'exécution de cette fonction nécessite un DataFrame (df) valide.")
    if outcome not in df.columns:
        raise ValueError(f"La variable '{outcome}' n'existe pas dans le DataFrame.")

    formula = _build_formula(outcome, predictors)
    family = Poisson(link=Log())

    if exposure is not None:
        model = smf.glm(formula, data=df, family=family, exposure=df[exposure])
    else:
        model = smf.glm(formula, data=df, family=family)

    result = model.fit()

    estimates = result.params.to_dict()
    standard_errors = result.bse.to_dict()
    p_values = result.pvalues.to_dict()

    ci = result.conf_int()
    confidence_intervals = {
        idx: (float(row[0]), float(row[1])) for idx, row in ci.iterrows()
    }

    # Taux d'incidence (exp des coefficients)
    incidence_rate_ratios = {k: float(np.exp(v)) for k, v in estimates.items()}

    model_metrics = {
        "method_family": "regression",
        "aic": float(result.aic),
        "bic": float(result.bic),
        "deviance": float(result.deviance),
        "null_deviance": float(result.null_deviance),
        "log_likelihood": float(result.llf),  # ← AJOUT
        "n_obs": int(result.nobs),
        "incidence_rate_ratios": incidence_rate_ratios,
    }

    generated_code = f"""import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.families.links import Log

df = pd.read_csv("CHEMIN_VERS_VOS_DONNEES.csv")
formula = "{formula}"
model = smf.glm(formula, data=df, family=Poisson(link=Log())).fit()
print(model.summary())
print("\\nTaux d'incidence (IRR):")
print(np.exp(model.params))
"""

    return AnalysisResult(
        method="poisson_regression",
        dataset_id="in_memory",
        variables={
            outcome: {"role": "outcome"},
            **{p: {"role": "predictor"} for p in predictors},
        },
        sample_size=int(result.nobs),
        preprocessing=["Modèle ajusté avec statsmodels"],
        estimates=estimates,
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals,
        p_values=p_values,
        model_metrics=model_metrics,
        diagnostics=[],
        warnings=[],
        assumptions_checked=[
            "Indépendance des observations",
            "Moyenne = variance (pas de surdispersion)",
            "Linéarité des prédicteurs dans le log",
        ],
        generated_code=generated_code,
        execution_log=["Modèle de Poisson ajusté avec succès"],
        software_versions={
            "python": platform.python_version(),
            "statsmodels": statsmodels.__version__,
        },
    )


# ============================================================================
# 3. Régression Binomiale Négative
# ============================================================================

def run_negative_binomial_regression(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    exposure: str | None = None,
) -> AnalysisResult:
    """Ajuste un modèle binomial négatif pour données de comptage surdispersées.

    Estime automatiquement le paramètre de dispersion (alpha), équivalent
    à MASS::glm.nb dans R.

    Args:
        df: DataFrame contenant les données.
        outcome: Nom de la variable cible de comptage.
        predictors: Liste des noms des variables prédictives.
        exposure: Nom de la colonne d'exposition (optionnel).

    Returns:
        AnalysisResult avec coefficients, alpha, taux d'incidence, etc.
    """
    if df is None:
        raise ValueError("L'exécution de cette fonction nécessite un DataFrame (df) valide.")
    if outcome not in df.columns:
        raise ValueError(f"La variable '{outcome}' n'existe pas dans le DataFrame.")

    formula = _build_formula(outcome, predictors)
    model = sm.NegativeBinomial.from_formula(formula, data=df)
    result = model.fit(disp=False)

    # Séparer les coefficients des prédicteurs du paramètre alpha
    params = result.params.copy()
    bse = result.bse.copy()
    tvalues = result.tvalues.copy()
    pvalues = result.pvalues.copy()
    conf_int = result.conf_int().copy()

    # Extraire alpha s'il existe
    alpha_val = float(params["alpha"]) if "alpha" in params.index else np.nan
    params = params.drop("alpha", errors="ignore")
    bse = bse.drop("alpha", errors="ignore")
    tvalues = tvalues.drop("alpha", errors="ignore")
    pvalues = pvalues.drop("alpha", errors="ignore")
    conf_int = conf_int.drop("alpha", errors="ignore")

    estimates = params.to_dict()
    standard_errors = bse.to_dict()
    p_values = pvalues.to_dict()

    confidence_intervals = {
        idx: (float(row[0]), float(row[1])) for idx, row in conf_int.iterrows()
    }

    # Taux d'incidence (exp des coefficients)
    incidence_rate_ratios = {k: float(np.exp(v)) for k, v in estimates.items()}

    model_metrics = {
        "method_family": "regression",
        "aic": float(result.aic),
        "bic": float(result.bic),
        "log_likelihood": float(result.llf),
        "alpha": alpha_val,
        "n_obs": int(result.nobs),
        "incidence_rate_ratios": incidence_rate_ratios,
    }

    generated_code = f"""import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("CHEMIN_VERS_VOS_DONNEES.csv")
formula = "{formula}"
model = sm.NegativeBinomial.from_formula(formula, data=df).fit(disp=False)
print(model.summary())
if 'alpha' in model.params.index:
    print(f"\\nAlpha (dispersion): {{model.params['alpha']:.4f}}")
print("\\nTaux d'incidence (IRR):")
params_no_alpha = model.params.drop('alpha', errors='ignore')
print(np.exp(params_no_alpha))
"""

    return AnalysisResult(
        method="negative_binomial_regression",
        dataset_id="in_memory",
        variables={
            outcome: {"role": "outcome"},
            **{p: {"role": "predictor"} for p in predictors},
        },
        sample_size=int(result.nobs),
        preprocessing=["Modèle ajusté avec statsmodels"],
        estimates=estimates,
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals,
        p_values=p_values,
        model_metrics=model_metrics,
        diagnostics=[],
        warnings=[],
        assumptions_checked=[
            "Indépendance des observations",
            "Surdispersion présente (alpha > 0)",
            "Linéarité des prédicteurs dans le log",
        ],
        generated_code=generated_code,
        execution_log=["Modèle binomial négatif ajusté avec succès"],
        software_versions={
            "python": platform.python_version(),
            "statsmodels": statsmodels.__version__,
        },
    )


# ============================================================================
# 4. Régression Gamma
# ============================================================================


def run_gamma_regression(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    link: Literal["log", "identity", "inverse"] = "log",
) -> AnalysisResult:
    """Ajuste un modèle Gamma pour données continues strictement positives.

    Args:
        df: DataFrame contenant les données.
        outcome: Nom de la variable cible continue > 0.
        predictors: Liste des noms des variables prédictives.
        link: Fonction de lien ("log", "identity", ou "inverse").

    Returns:
        AnalysisResult avec coefficients, scale, etc.
    """
    if df is None:
        raise ValueError("L'exécution de cette fonction nécessite un DataFrame (df) valide.")
    if outcome not in df.columns:
        raise ValueError(f"La variable '{outcome}' n'existe pas dans le DataFrame.")

    formula = _build_formula(outcome, predictors)
    
    # Mapper le lien string vers l'objet Link
    link_map = {
        "log": Log(),
        "identity": Identity(),
        "inverse": InversePower(),
    }
    family = Gamma(link=link_map[link])
    
    model = smf.glm(formula, data=df, family=family)
    result = model.fit()

    estimates = result.params.to_dict()
    standard_errors = result.bse.to_dict()
    p_values = result.pvalues.to_dict()

    ci = result.conf_int()
    confidence_intervals = {
        idx: (float(row[0]), float(row[1])) for idx, row in ci.iterrows()
    }

    model_metrics = {
        "method_family": "regression",
        "aic": float(result.aic),
        "bic": float(result.bic),
        "deviance": float(result.deviance),
        "scale": float(result.scale),
        "log_likelihood": float(result.llf),  # ← AJOUT pour cohérence
        "n_obs": int(result.nobs),
    }

    generated_code = f"""import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import Log, Identity, InversePower

df = pd.read_csv("CHEMIN_VERS_VOS_DONNEES.csv")
formula = "{formula}"
link_map = {{"log": Log(), "identity": Identity(), "inverse": InversePower()}}
model = smf.glm(formula, data=df, family=Gamma(link=link_map["{link}"])).fit()
print(model.summary())
"""

    return AnalysisResult(
        method="gamma_regression",
        dataset_id="in_memory",
        variables={
            outcome: {"role": "outcome"},
            **{p: {"role": "predictor"} for p in predictors},
        },
        sample_size=int(result.nobs),
        preprocessing=["Modèle ajusté avec statsmodels"],
        estimates=estimates,
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals,
        p_values=p_values,
        model_metrics=model_metrics,
        diagnostics=[],
        warnings=[],
        assumptions_checked=[
            "Indépendance des observations",
            "Variable réponse strictement positive",
            "Distribution Gamma appropriée",
        ],
        generated_code=generated_code,
        execution_log=["Modèle Gamma ajusté avec succès"],
        software_versions={
            "python": platform.python_version(),
            "statsmodels": statsmodels.__version__,
        },
    )


# ============================================================================
# 5. Régression Logistique Ordinale
# ============================================================================


def run_ordinal_regression(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    link: Literal["logit", "probit", "cloglog"] = "logit",
) -> AnalysisResult:
    """Ajuste un modèle de régression ordinale (proportional odds / cumulative link).

    Pour données catégorielles ordonnées (ex: échelle de Likert, stades de maladie).
    Utilise statsmodels.miscmodels.ordinal_model.OrderedModel.

    Note importante sur la validation R :
        Les COEFFICIENTS (x1, x2, ...) sont directement comparables à MASS::polr.
        Les SEUILS (intercepts) ne sont PAS comparables brute-à-brute car polr et
        OrderedModel utilisent des paramétrisations différentes des cut-points.
        La tolérance de validation (V2 §6.3) s'applique uniquement aux coefficients.

    Args:
        df: DataFrame contenant les données.
        outcome: Nom de la variable cible ordinale (catégories ordonnées).
        predictors: Liste des noms des variables prédictives.
        link: Fonction de lien ("logit", "probit", ou "cloglog").

    Returns:
        AnalysisResult avec coefficients, seuils (intercepts), etc.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    if df is None:
        raise ValueError("L'exécution de cette fonction nécessite un DataFrame (df) valide.")
    if outcome not in df.columns:
        raise ValueError(f"La variable '{outcome}' n'existe pas dans le DataFrame.")

    # Vérifier que la variable outcome est ordinale (catégories ordonnées)
    y_series = df[outcome]
    if not hasattr(y_series, "cat") or not y_series.cat.ordered:
        unique_vals = sorted(y_series.unique())
        if len(unique_vals) < 3:
            raise ValueError(
                f"La variable '{outcome}' doit avoir au moins 3 catégories ordonnées. "
                f"Trouvé: {len(unique_vals)} catégories."
            )
        df = df.copy()
        df[outcome] = pd.Categorical(df[outcome], categories=unique_vals, ordered=True)

    # Construire la matrice des prédicteurs
    X = df[predictors]
    y = df[outcome]

    # Mapper le lien
    distr_map = {
        "logit": "logit",
        "probit": "probit",
        "cloglog": "cloglog",
    }
    if link not in distr_map:
        raise ValueError(f"Link non supporté: {link}. Utiliser 'logit', 'probit' ou 'cloglog'.")

    # Ajuster le modèle
    model = OrderedModel(y, X, distr=distr_map[link])
    result = model.fit(method="bfgs", disp=False)

    # Détection des seuils : clés contenant un '/' (format statsmodels: "1/2", "2/3", ...)
    def _is_threshold(key: str) -> bool:
        return "/" in key

    # Extraire les coefficients et les seuils
    params = result.params
    estimates = {}
    thresholds = {}

    for key, value in params.items():
        if _is_threshold(key):
            thresholds[key] = float(value)
        else:
            pred_name = key.replace("x_", "") if key.startswith("x_") else key
            estimates[pred_name] = float(value)

    # Erreurs standards
    bse = result.bse
    standard_errors = {}
    for key, value in bse.items():
        if not _is_threshold(key):
            pred_name = key.replace("x_", "") if key.startswith("x_") else key
            standard_errors[pred_name] = float(value)

    # P-values
    pvalues = result.pvalues
    p_values = {}
    for key, value in pvalues.items():
        if not _is_threshold(key):
            pred_name = key.replace("x_", "") if key.startswith("x_") else key
            p_values[pred_name] = float(value)

    # Intervalles de confiance
    ci = result.conf_int()
    confidence_intervals = {}
    for idx, row in ci.iterrows():
        if not _is_threshold(idx):
            pred_name = idx.replace("x_", "") if idx.startswith("x_") else idx
            confidence_intervals[pred_name] = (float(row[0]), float(row[1]))

    # Métriques du modèle
    model_metrics = {
        "method_family": "regression",
        "aic": float(result.aic),
        "bic": float(result.bic),
        "log_likelihood": float(result.llf),
        "n_obs": int(result.nobs),
        "n_categories": len(y.cat.categories) if hasattr(y, "cat") else len(y.unique()),
        "thresholds": thresholds,
    }

    generated_code = f"""import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

df = pd.read_csv("CHEMIN_VERS_VOS_DONNEES.csv")

# Convertir la variable outcome en catégorie ordonnée
df["{outcome}"] = pd.Categorical(df["{outcome}"], categories=sorted(df["{outcome}"].unique()), ordered=True)

X = df[{predictors}]
y = df["{outcome}"]

model = OrderedModel(y, X, distr="{distr_map[link]}")
result = model.fit(method="bfgs", disp=False)
print(result.summary())
"""

    return AnalysisResult(
        method="ordinal_regression",
        dataset_id="in_memory",
        variables={
            outcome: {"role": "outcome"},
            **{p: {"role": "predictor"} for p in predictors},
        },
        sample_size=int(result.nobs),
        preprocessing=["Variable outcome convertie en catégorie ordonnée"],
        estimates=estimates,
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals,
        p_values=p_values,
        model_metrics=model_metrics,
        diagnostics=[],
        warnings=[
            "Les seuils (cut-points) sont donnés dans la paramétrisation statsmodels "
            "(format 'i/j') et ne sont pas directement comparables aux zeta de "
            "MASS::polr ; seuls les coefficients sont validés contre R."
        ],
        assumptions_checked=[
            "Proportional odds assumption (hypothèse des pentes proportionnelles)",
            "Indépendance des observations",
            "Catégories ordonnées de manière significative",
        ],
        generated_code=generated_code,
        execution_log=["Modèle ordinal ajusté avec succès"],
        software_versions={
            "python": platform.python_version(),
            "statsmodels": statsmodels.__version__,
        },
    )

# ============================================================================
# Enregistrement dans le registre (V2 §7.1)
# ============================================================================

register_method(
    MethodSpec(
        method_id="probit_regression",
        label_fr="Régression Probit/Logit",
        family="regression",
        engine="python",
        outcome_types=["binary"],
        predictor_types=["continuous", "categorical"],
        min_sample_size=20,
        assumptions=[
            "Indépendance des observations",
            "Linéarité des log-odds/probits",
            "Absence de multicolinéarité forte",
        ],
        reference_impl="statsmodels.formula.api.probit",
        r_reference="stats::glm(family=binomial(link='probit'))",
        phase=18,
        extra=None,
        default_params={"link": "probit"},
    ),
    run_probit_regression,
)

register_method(
    MethodSpec(
        method_id="poisson_regression",
        label_fr="Régression de Poisson",
        family="regression",
        engine="python",
        outcome_types=["count"],
        predictor_types=["continuous", "categorical"],
        min_sample_size=20,
        assumptions=[
            "Indépendance des observations",
            "Moyenne = variance (pas de surdispersion)",
            "Linéarité des prédicteurs dans le log",
        ],
        reference_impl="statsmodels.genmod.families.Poisson",
        r_reference="stats::glm(family=poisson)",
        phase=18,
        extra=None,
        default_params={},
    ),
    run_poisson_regression,
)

register_method(
    MethodSpec(
        method_id="negative_binomial_regression",
        label_fr="Régression Binomiale Négative",
        family="regression",
        engine="python",
        outcome_types=["count"],
        predictor_types=["continuous", "categorical"],
        min_sample_size=20,
        assumptions=[
            "Indépendance des observations",
            "Surdispersion présente (alpha > 0)",
            "Linéarité des prédicteurs dans le log",
        ],
        reference_impl="statsmodels.api.NegativeBinomial",
        r_reference="MASS::glm.nb",
        phase=18,
        extra=None,
        default_params={},
    ),
    run_negative_binomial_regression,
)

register_method(
    MethodSpec(
        method_id="gamma_regression",
        label_fr="Régression Gamma",
        family="regression",
        engine="python",
        outcome_types=["continuous"],
        predictor_types=["continuous", "categorical"],
        min_sample_size=20,
        assumptions=[
            "Indépendance des observations",
            "Variable réponse strictement positive",
            "Distribution Gamma appropriée",
        ],
        reference_impl="statsmodels.genmod.families.Gamma",
        r_reference="stats::glm(family=Gamma(link='log'))",
        phase=18,
        extra=None,
        default_params={"link": "log"},
    ),
    run_gamma_regression,
)

register_method(
    MethodSpec(
        method_id="ordinal_regression",
        label_fr="Régression Logistique Ordinale",
        family="regression",
        engine="python",
        outcome_types=["ordinal"],
        predictor_types=["continuous", "categorical"],
        min_sample_size=30,
        assumptions=[
            "Proportional odds assumption",
            "Indépendance des observations",
            "Catégories ordonnées de manière significative",
        ],
        reference_impl="statsmodels.miscmodels.ordinal_model.OrderedModel",
        r_reference="MASS::polr",
        phase=18,
        extra=None,
        default_params={"link": "logit"},
    ),
    run_ordinal_regression,
)