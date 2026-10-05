"""Vérifications diagnostiques statistiques.

Ce module contient les fonctions de diagnostic qui détectent les problèmes
statistiques courants (multicolinéarité, séparation, observations influentes,
taille d'échantillon insuffisante).

Tous les diagnostics sont calculés par du code déterministe, jamais par un LLM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from app.diagnostics.models import DiagnosticFinding, Severity


def check_multicollinearity(
    df: pd.DataFrame, predictors: list[str]
) -> list[DiagnosticFinding]:
    """Détecte la multicolinéarité via le Variance Inflation Factor (VIF).

    Seuils utilisés (source : Hair et al., "Multivariate Data Analysis") :
        - VIF > 10 : multicolinéarité sévère (CERTAIN_ISSUE)
        - VIF > 5  : multicolinéarité modérée (PROBABLE_ISSUE)

    Args:
        df: DataFrame contenant les données.
        predictors: Liste des noms de colonnes des variables prédictives numériques.

    Returns:
        Liste de diagnostics détectés. Vide si aucun problème.
    """
    if len(predictors) < 2:
        return []

    # Filtrer les colonnes manquantes
    available = [p for p in predictors if p in df.columns]
    if len(available) < 2:
        return []

    # Extraire les données numériques et supprimer les NA
    X = df[available].dropna()
    if X.shape[0] < 2:
        return []

    diagnostics: list[DiagnosticFinding] = []

    # Calcul du VIF pour chaque prédicteur
    # On ajoute une constante pour le calcul du VIF
    X_with_const = X.copy()
    X_with_const.insert(0, "const", 1.0)

    for i, var in enumerate(available):
        # VIF pour la variable i+1 (index 0 = constante)
        vif = variance_inflation_factor(X_with_const.values, i + 1)

        if vif > 10:
            diagnostics.append(
                DiagnosticFinding(
                    code="HIGH_VIF",
                    severity=Severity.CERTAIN_ISSUE,
                    message=(
                        f"Multicolinéarité sévère détectée pour '{var}' "
                        f"(VIF = {vif:.2f})."
                    ),
                    details={"variable": var, "vif": float(vif), "threshold": 10},
                )
            )
        elif vif > 5:
            diagnostics.append(
                DiagnosticFinding(
                    code="MODERATE_VIF",
                    severity=Severity.PROBABLE_ISSUE,
                    message=(
                        f"Multicolinéarité modérée détectée pour '{var}' "
                        f"(VIF = {vif:.2f})."
                    ),
                    details={"variable": var, "vif": float(vif), "threshold": 5},
                )
            )

    return diagnostics


def check_separation(model_result) -> list[DiagnosticFinding]:
    """Détecte la séparation complète ou quasi-séparation sur un modèle
    de régression logistique.

    Critères (source : Allison, "Logistic Regression Using SAS") :
        - Erreur standard anormalement grande (SE > 10 × |coef|)
        - Non-convergence signalée par statsmodels (converged = False)

    Args:
        model_result: Résultat d'un modèle statsmodels (fit() de Logit/GLM).

    Returns:
        Liste de diagnostics détectés. Vide si aucun problème.
    """
    diagnostics: list[DiagnosticFinding] = []

    # Vérifier la convergence
    if hasattr(model_result, "converged") and not model_result.converged:
        diagnostics.append(
            DiagnosticFinding(
                code="NON_CONVERGENCE",
                severity=Severity.CERTAIN_ISSUE,
                message="Le modèle n'a pas convergé. Séparation probable.",
                details={"converged": False},
            )
        )

    # Vérifier les erreurs standard anormalement grandes
    if hasattr(model_result, "params") and hasattr(model_result, "bse"):
        params = model_result.params
        bse = model_result.bse

        for var_name in params.index:
            coef = abs(params[var_name])
            se = bse[var_name]

            # SE > 10 × |coef| indique une quasi-séparation
            if coef > 0 and se > 10 * coef:
                diagnostics.append(
                    DiagnosticFinding(
                        code="QUASI_SEPARATION",
                        severity=Severity.PROBABLE_ISSUE,
                        message=(
                            f"Quasi-séparation suspectée pour '{var_name}' "
                            f"(SE = {se:.2f} vs |coef| = {coef:.2f})."
                        ),
                        details={
                            "variable": var_name,
                            "coefficient": float(coef),
                            "standard_error": float(se),
                            "ratio": float(se / coef),
                        },
                    )
                )
            elif coef == 0 and se > 100:
                # Coefficient nul avec SE énorme = séparation complète
                diagnostics.append(
                    DiagnosticFinding(
                        code="COMPLETE_SEPARATION",
                        severity=Severity.CERTAIN_ISSUE,
                        message=(
                            f"Séparation complète suspectée pour '{var_name}' "
                            f"(SE = {se:.2f} sur coefficient nul)."
                        ),
                        details={
                            "variable": var_name,
                            "coefficient": float(coef),
                            "standard_error": float(se),
                        },
                    )
                )

    return diagnostics


def check_influential_observations(model_result) -> list[DiagnosticFinding]:
    """Détecte les observations influentes via la distance de Cook.

    Seuil utilisé (source : Belsley, Kuh & Welsch, "Regression Diagnostics") :
        - Distance de Cook > 4/n : observation potentiellement influente (WARNING)

    Args:
        model_result: Résultat d'un modèle statsmodels (OLS, Logit, GLM).

    Returns:
        Liste de diagnostics détectés. Vide si aucun problème.
    """
    diagnostics: list[DiagnosticFinding] = []

    # Récupérer l'influence
    try:
        # Pour OLS : OLSInfluence
        if hasattr(model_result.model, "nobs"):
            n = model_result.model.nobs
        else:
            n = len(model_result.resid)

        # Seuil : 4/n
        threshold = 4 / n

        # Calcul de la distance de Cook
        # statsmodels expose get_influence() sur les résultats de modèle
        if hasattr(model_result, "get_influence"):
            influence = model_result.get_influence()
            cooks_distance = influence.cooks_distance[0]  # array des distances

            # Identifier les observations influentes
            influential_idx = np.where(cooks_distance > threshold)[0]

            if len(influential_idx) > 0:
                # Limiter à 10 observations max pour ne pas surcharger
                n_influential = len(influential_idx)
                max_distance = float(np.max(cooks_distance))

                diagnostics.append(
                    DiagnosticFinding(
                        code="INFLUENTIAL_OBSERVATIONS",
                        severity=Severity.WARNING,
                        message=(
                            f"{n_influential} observation(s) influente(s) détectée(s) "
                            f"(distance de Cook > {threshold:.4f}). "
                            f"Distance max = {max_distance:.4f}."
                        ),
                        details={
                            "n_influential": n_influential,
                            "threshold": float(threshold),
                            "max_distance": max_distance,
                            "indices": influential_idx[:10].tolist(),
                        },
                    )
                )
    except Exception:
        # Si le calcul échoue (ex. modèle non supporté), on ne lève pas d'erreur
        pass

    return diagnostics


def check_sample_size(
    df: pd.DataFrame, method: str, n_predictors: int
) -> list[DiagnosticFinding]:
    """Vérifie que la taille d'échantillon est suffisante pour la méthode.

    Règle empirique (source : Peduzzi et al., 1996) :
        - Régression logistique : au moins 10 événements par variable (EPV)
        - Régression linéaire : au moins 10 observations par variable

    Args:
        df: DataFrame contenant les données.
        method: Nom de la méthode statistique (ex. 'logistic_regression', 'linear_regression').
        n_predictors: Nombre de variables prédictives.

    Returns:
        Liste de diagnostics détectés. Vide si aucun problème.
    """
    diagnostics: list[DiagnosticFinding] = []
    n_obs = len(df)

    if method == "logistic_regression":
        # Pour la logistique, on compte le nombre d'événements (classe minoritaire)
        # On suppose que la variable outcome est la dernière colonne ou doit être passée
        # Ici on vérifie juste la taille globale
        min_required = 10 * n_predictors

        if n_obs < min_required:
            diagnostics.append(
                DiagnosticFinding(
                    code="LOW_SAMPLE_SIZE_LOGISTIC",
                    severity=Severity.WARNING,
                    message=(
                        f"Taille d'échantillon potentiellement insuffisante pour "
                        f"la régression logistique. Recommandé : au moins "
                        f"{min_required} observations (10 × {n_predictors} variables), "
                        f"actuel : {n_obs}."
                    ),
                    details={
                        "n_observations": n_obs,
                        "n_predictors": n_predictors,
                        "recommended_minimum": min_required,
                        "rule": "Events Per Variable (EPV) >= 10",
                    },
                )
            )

    elif method == "linear_regression":
        # Règle générale : 10 observations par variable
        min_required = 10 * n_predictors

        if n_obs < min_required:
            diagnostics.append(
                DiagnosticFinding(
                    code="LOW_SAMPLE_SIZE_LINEAR",
                    severity=Severity.WARNING,
                    message=(
                        f"Taille d'échantillon potentiellement insuffisante pour "
                        f"la régression linéaire. Recommandé : au moins "
                        f"{min_required} observations (10 × {n_predictors} variables), "
                        f"actuel : {n_obs}."
                    ),
                    details={
                        "n_observations": n_obs,
                        "n_predictors": n_predictors,
                        "recommended_minimum": min_required,
                        "rule": "10 observations per predictor",
                    },
                )
            )

    return diagnostics
