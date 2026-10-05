"""Génération de plans d'analyse structurés."""

from __future__ import annotations

from app.agent.llm_provider import LLMProvider, _extract_json
from app.data.models import DataDictionary
from app.planner.models import AnalysisPlan


_PLANNER_SYSTEM_PROMPT = """Tu es un expert en méthodologie de recherche et en statistiques.

Ton rôle : proposer un plan d'analyse structuré à partir d'une question de recherche et d'un dictionnaire de données.

MÉTHODES SUPPORTÉES (tu dois choisir parmi celles-ci) :
- linear_regression : régression linéaire (outcome numérique continu)
- logistic_regression : régression logistique binaire (outcome binaire 0/1)
- multinomial_logistic_regression : régression logistique multinomiale (outcome catégoriel >= 3 modalités)
- chi_square : test du chi-carré d'indépendance (deux variables catégorielles)
- t_test : test t de Student (comparaison de 2 groupes, outcome numérique)
- anova : analyse de variance (comparaison de >= 3 groupes, outcome numérique)

DESIGNS D'ÉTUDE SUPPORTÉS :
- cross_sectional : étude transversale (description, association)
- comparative : comparaison de groupes
- associative : recherche d'association entre variables
- predictive : modèle prédictif

RÈGLES STRICTES :
1. Si la question est ambiguë ou si les variables mentionnées n'existent pas dans le dictionnaire de données, retourne confidence='low' plutôt qu'un plan hasardeux.
2. Le nom des variables (outcome_variable, predictor_variables) doit correspondre EXACTEMENT aux noms dans le dictionnaire de données.
3. assumptions_to_check doit lister les hypothèses statistiques pertinentes pour la méthode choisie.
4. rationale doit expliquer brièvement pourquoi cette méthode est appropriée.
5. Tu dois répondre UNIQUEMENT avec un objet JSON valide, sans aucun texte additionnel, sans markdown, sans explication. Le format de sortie doit être strictement du JSON.

Exemple de réponse attendue (format JSON pur) :
{
  "research_question": "Quel est l'effet de X sur Y ?",
  "outcome_variable": "Y",
  "predictor_variables": ["X", "Z"],
  "study_design": "predictive",
  "proposed_method": "linear_regression",
  "reference_category": null,
  "assumptions_to_check": ["Linéarité", "Normalité des résidus", "Homoscédasticité"],
  "rationale": "La variable outcome Y est numérique continue, une régression linéaire est appropriée.",
  "confidence": "high"
}
"""


def build_analysis_plan(
    question: str,
    data_dictionary: DataDictionary,
    llm: LLMProvider,
) -> AnalysisPlan:
    """Génère un plan d'analyse structuré à partir d'une question et d'un dictionnaire de données."""
    variables_summary = []
    for var_name, var in data_dictionary.variables.items():
        role_hint = ""
        if var.semantic_type:
            role_hint = f" ({var.semantic_type}"
            if var.levels:
                role_hint += f", modalités: {var.levels[:5]}"
            role_hint += ")"
        variables_summary.append(f"- {var_name}{role_hint}")

    context = f"""Question de recherche : {question}

Dictionnaire de données (dataset '{data_dictionary.dataset_name}', {data_dictionary.n_rows} lignes, {data_dictionary.n_columns} colonnes) :
{chr(10).join(variables_summary)}

Propose un plan d'analyse structuré en respectant le schéma JSON attendu."""

    messages = [
        {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]

    response = llm.complete(
        messages=messages,
        response_schema=AnalysisPlan,
    )

    if response.structured_output and isinstance(response.structured_output, AnalysisPlan):
        return response.structured_output

    if response.text:
        try:
            clean_text = _extract_json(response.text)
            return AnalysisPlan.model_validate_json(clean_text)
        except Exception as e:
            # Affiche les 300 premiers caractères de la réponse brute pour le débogage
            raise ValueError(
                f"Le LLM n'a pas retourné un plan d'analyse valide. Erreur : {e}. "
                f"Texte brut reçu (début) : {response.text[:300]}"
            ) from e

    raise ValueError("Le LLM n'a retourné ni texte ni output structuré.")


def needs_clarification(plan: AnalysisPlan) -> bool:
    """Détermine si le plan nécessite une clarification utilisateur."""
    return plan.confidence == "low"
