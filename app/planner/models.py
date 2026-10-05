"""Modèles Pydantic pour les plans d'analyse.

Ce module définit la structure AnalysisPlan, objet structuré produit par
l'Analysis Planner à partir d'une question utilisateur et d'un dictionnaire
de données.
"""

from typing import Literal

from pydantic import BaseModel, Field


class AnalysisPlan(BaseModel):
    """Plan d'analyse structuré produit par l'Analysis Planner — jamais du
    texte libre, toujours validé par ce schéma Pydantic."""

    research_question: str = Field(
        description="Question de recherche reformulée de manière précise"
    )
    outcome_variable: str = Field(
        description="Nom de la variable cible (outcome)"
    )
    predictor_variables: list[str] = Field(
        description="Liste des noms des variables prédictives"
    )
    study_design: Literal[
        "cross_sectional", "comparative", "associative", "predictive"
    ] = Field(description="Type de plan d'étude")
    proposed_method: str = Field(
        description="Méthode statistique proposée (ex: 'linear_regression')"
    )
    reference_category: str | None = Field(
        default=None,
        description="Catégorie de référence pour les variables catégorielles (optionnel)",
    )
    assumptions_to_check: list[str] = Field(
        description="Liste des hypothèses statistiques à vérifier"
    )
    rationale: str = Field(
        description="Justification du choix de la méthode"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Niveau de confiance dans le plan (low = clarification nécessaire)"
    )
