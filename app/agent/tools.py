"""Wrappers exposant les fonctions statistiques au tool calling du LLM.

Ce module est le SEUL point d'entrée par lequel l'AI Layer déclenche un calcul
statistique. Il ne reçoit jamais de DataFrame brut depuis le LLM — uniquement
des noms de variables et des arguments structurés.

Contraintes respectées :
- T8 : aucun DataFrame brut envoyé au LLM
- T9 : passage par execute_tool, jamais d'appel direct aux fonctions stats
- T14 : confirmation humaine pour les actions modifiant l'état (gérée par l'orchestrator)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.results.models import AnalysisResult


# ──────────────────────────────────────────────────────────────────────
# Schémas JSON des outils exposés au LLM (tool calling OpenAI-compatible)
# ──────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    # ── Tests classiques (Phase 3) ───────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "run_chi_square",
            "description": (
                "Test du chi-carré d'indépendance entre deux variables catégorielles. "
                "Utilise scipy.stats.chi2_contingency."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "var1": {
                        "type": "string",
                        "description": "Nom de la première variable catégorielle",
                    },
                    "var2": {
                        "type": "string",
                        "description": "Nom de la deuxième variable catégorielle",
                    },
                },
                "required": ["var1", "var2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_t_test",
            "description": (
                "Test t de Student pour comparer les moyennes d'une variable numérique "
                "entre 2 groupes. Le groupe doit avoir exactement 2 modalités."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "description": "Nom de la variable numérique (outcome)",
                    },
                    "group": {
                        "type": "string",
                        "description": (
                            "Nom de la variable catégorielle à 2 modalités (groupes)"
                        ),
                    },
                },
                "required": ["outcome", "group"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_anova",
            "description": (
                "Analyse de variance (ANOVA) pour comparer les moyennes d'une variable "
                "numérique entre 3 groupes ou plus."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "description": "Nom de la variable numérique (outcome)",
                    },
                    "group": {
                        "type": "string",
                        "description": "Nom de la variable catégorielle (groupes)",
                    },
                },
                "required": ["outcome", "group"],
            },
        },
    },
    # ── Régressions (Phases 3 et 6) ───────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "run_linear_regression",
            "description": (
                "Régression linéaire multiple (OLS) via statsmodels.formula.api.ols. "
                "Outcome numérique continu, prédicteurs numériques ou catégoriels."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "description": "Nom de la variable numérique (outcome)",
                    },
                    "predictors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Liste des noms des variables prédictives",
                    },
                },
                "required": ["outcome", "predictors"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_logistic_regression",
            "description": (
                "Régression logistique binaire via statsmodels.api.Logit. "
                "Outcome binaire (0/1), prédicteurs numériques ou catégoriels."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "description": "Nom de la variable binaire (outcome, 0/1)",
                    },
                    "predictors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Liste des noms des variables prédictives",
                    },
                    "reference_category": {
                        "type": "string",
                        "description": (
                            "Catégorie de référence pour les prédicteurs catégoriels "
                            "(optionnel)"
                        ),
                    },
                },
                "required": ["outcome", "predictors"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_multinomial_logistic_regression",
            "description": (
                "Régression logistique multinomiale via statsmodels.api.MNLogit. "
                "Outcome catégoriel à 3 modalités ou plus."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "description": (
                            "Nom de la variable catégorielle (outcome, >= 3 modalités)"
                        ),
                    },
                    "predictors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Liste des noms des variables prédictives",
                    },
                },
                "required": ["outcome", "predictors"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_firth_logistic",
            "description": (
                "Régression logistique de Firth (correction de biais pour séparation "
                "complète/quasi-séparation). Tente firthlogist (Python), repli sur R."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "description": "Nom de la variable binaire (outcome, 0/1)",
                    },
                    "predictors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Liste des noms des variables prédictives",
                    },
                },
                "required": ["outcome", "predictors"],
            },
        },
    },
    # ── Diagnostics (Phase 5) ────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "check_multicollinearity",
            "description": (
                "Vérifie la multicolinéarité entre prédicteurs via le VIF. "
                "Seuils : VIF > 10 → CERTAIN_ISSUE, VIF > 5 → PROBABLE_ISSUE."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "predictors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Liste des noms des variables prédictives",
                    },
                },
                "required": ["predictors"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_sample_size",
            "description": (
                "Vérifie la taille d'échantillon selon la règle EPV (Events Per Variable). "
                "Au moins 10 événements par variable prédicteur pour la régression logistique."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": (
                            "Méthode statistique ('logistic_regression', "
                            "'linear_regression', etc.)"
                        ),
                    },
                    "n_predictors": {
                        "type": "integer",
                        "description": "Nombre de variables prédictives",
                    },
                },
                "required": ["method", "n_predictors"],
            },
        },
    },
    # ── Rapports (Phase 4) ──────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": (
                "Génère un rapport à partir d'un AnalysisResult. "
                "Formats supportés : HTML, Markdown."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "output_format": {
                        "type": "string",
                        "enum": ["html", "markdown"],
                        "description": "Format de sortie du rapport.",
                    },
                },
                "required": ["output_format"],
            },
        },
    },
]


# ──────────────────────────────────────────────────────────────────────
# Point d'entrée unique pour l'exécution d'outils
# ──────────────────────────────────────────────────────────────────────

def execute_tool(
    name: str,
    arguments: dict[str, Any],
    df: pd.DataFrame | None = None,
    result: AnalysisResult | None = None,
    data_dictionary: Any | None = None,
) -> AnalysisResult | str | list:
    """Route vers la bonne fonction selon le nom de l'outil.

    C'est le SEUL point d'entrée par lequel l'AI Layer déclenche un calcul.
    Le LLM ne manipule que des noms de variables, jamais les données elles-mêmes.

    Args:
        name: Nom de l'outil (ex: 'run_linear_regression', 'generate_report').
        arguments: Dictionnaire des arguments structurés selon TOOL_SCHEMAS.
        df: DataFrame contenant les données (requis pour les outils statistiques).
        result: AnalysisResult (requis pour generate_report).
        data_dictionary: Dictionnaire de données (optionnel, pour contexte).

    Returns:
        AnalysisResult pour les méthodes statistiques,
        str pour generate_report,
        list pour les checks de diagnostics.

    Raises:
        ValueError: Si l'outil est inconnu ou si les arguments requis sont manquants.
    """
    # ── Vérification de l'existence de l'outil ──────────────────────
    available_tools = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
    if name not in available_tools:
        raise ValueError(
            f"Outil '{name}' non reconnu. "
            f"Outils disponibles : {', '.join(available_tools)}"
        )

    # ── Outils statistiques : df obligatoire ────────────────────────
    statistical_tools = {
        "run_chi_square", "run_t_test", "run_anova",
        "run_linear_regression", "run_logistic_regression",
        "run_multinomial_logistic_regression", "run_firth_logistic",
        "check_multicollinearity", "check_sample_size",
    }
    if name in statistical_tools and df is None:
        raise ValueError(
            f"L'outil '{name}' nécessite un DataFrame (df) valide."
        )

    # ── generate_report : result obligatoire ────────────────────────
    if name == "generate_report":
        if result is None:
            raise ValueError(
                "L'outil 'generate_report' nécessite un AnalysisResult."
            )
        from app.reports.generator import generate_report
        output_format = arguments.get("output_format", "html")
        return generate_report(result=result, output_format=output_format)

    # ── Tests classiques (Phase 3) ─────────────────────────────────
    if name == "run_chi_square":
        from app.statistics.tests_classiques import run_chi_square

        return run_chi_square(
            df=df,
            var1=arguments["var1"],
            var2=arguments["var2"],
        )

    if name == "run_t_test":
        from app.statistics.tests_classiques import run_t_test

        return run_t_test(
            df=df,
            outcome=arguments["outcome"],
            group=arguments["group"],
        )

    if name == "run_anova":
        from app.statistics.tests_classiques import run_anova

        return run_anova(
            df=df,
            outcome=arguments["outcome"],
            group=arguments["group"],
        )

    # ── Régressions (Phases 3 et 6) ────────────────────────────────
    if name == "run_linear_regression":
        from app.statistics.regression import run_linear_regression

        return run_linear_regression(
            df=df,
            outcome=arguments["outcome"],
            predictors=arguments["predictors"],
        )

    if name == "run_logistic_regression":
        from app.statistics.regression import run_logistic_regression

        return run_logistic_regression(
            df=df,
            outcome=arguments["outcome"],
            predictors=arguments["predictors"],
            reference_category=arguments.get("reference_category"),
        )

    if name == "run_multinomial_logistic_regression":
        from app.statistics.regression import run_multinomial_logistic_regression

        return run_multinomial_logistic_regression(
            df=df,
            outcome=arguments["outcome"],
            predictors=arguments["predictors"],
        )

    if name == "run_firth_logistic":
        from app.statistics.regression import run_firth_logistic

        return run_firth_logistic(
            df=df,
            outcome=arguments["outcome"],
            predictors=arguments["predictors"],
        )

    # ── Diagnostics (Phase 5) ───────────────────────────────────────
    if name == "check_multicollinearity":
        from app.diagnostics.checks import check_multicollinearity

        return check_multicollinearity(
            df=df,
            predictors=arguments["predictors"],
        )

    if name == "check_sample_size":
        from app.diagnostics.checks import check_sample_size

        return check_sample_size(
            df=df,
            method=arguments["method"],
            n_predictors=arguments["n_predictors"],
        )

    # ── Fallback (ne devrait jamais être atteint) ───────────────────
    raise ValueError(
        f"Outil '{name}' non implémenté. "
        f"Outils disponibles : {', '.join(available_tools)}"
    )
