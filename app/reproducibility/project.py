"""Sérialisation et rejeu d'une analyse complète.

Ce module permet de :
- Sauvegarder une analyse (question + plan + résultat + code + versions) en JSON
- Recharger une analyse sauvegardée
- Rejouer l'analyse à l'identique via execute_tool (T9)

Contraintes respectées :
- T8 : aucun DataFrame brut dans le fichier projet
- T13 : code reproductible + seeds + versions logicielles
- T9 : replay via execute_tool, jamais d'appel direct aux fonctions stats
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.results.models import AnalysisResult

# ──────────────────────────────────────────────────────────────────────
# Modèle de projet d'analyse
# ──────────────────────────────────────────────────────────────────────


@dataclass
class AnalysisProject:
    """Projet d'analyse sérialisable.

    Contient tout le nécessaire pour rejouer une analyse à l'identique,
    sans jamais stocker les microdonnées brutes (T8).
    """

    # Métadonnées du projet
    project_id: str
    created_at: str  # ISO 8601
    dataset_name: str
    dataset_hash: str | None  # SHA256 du fichier source (si disponible)
    dataset_rows: int
    dataset_columns: int

    # Question de recherche
    research_question: str

    # Plan d'analyse
    proposed_method: str
    outcome_variable: str
    predictor_variables: list[str]
    reference_category: str | None
    study_design: str
    confidence: str
    rationale: str
    assumptions_to_check: list[str]

    # Résultat de l'analyse
    method: str
    sample_size: int
    estimates: dict[str, float]
    p_values: dict[str, float]
    model_metrics: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    warnings: list[str]
    assumptions_checked: list[str]
    generated_code: str
    execution_log: list[str]
    software_versions: dict[str, str]

    # Prétraitements appliqués
    preprocessing: list[dict[str, Any]] = field(default_factory=list)

    # Variables (métadonnées uniquement, pas de données)
    variables: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Chemin vers le fichier projet (populé après sauvegarde)
    project_path: str | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Sérialise le projet en dictionnaire (pour JSON)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisProject:
        """Désérialise un projet depuis un dictionnaire."""
        return cls(**data)


# ──────────────────────────────────────────────────────────────────────
# Construction d'un projet depuis une analyse exécutée
# ──────────────────────────────────────────────────────────────────────


def build_project_from_analysis(
    result: AnalysisResult,
    research_question: str,
    dataset_name: str,
    dataset_hash: str | None = None,
    dataset_rows: int = 0,
    dataset_columns: int = 0,
    plan: Any | None = None,
    project_id: str | None = None,
) -> AnalysisProject:
    """Construit un AnalysisProject depuis un AnalysisResult exécuté.

    Args:
        result: Résultat de l'analyse (après execute_tool).
        research_question: Question de recherche originale.
        dataset_name: Nom du fichier source.
        dataset_hash: Hash SHA256 du fichier source (optionnel).
        dataset_rows: Nombre de lignes du dataset.
        dataset_columns: Nombre de colonnes du dataset.
        plan: Plan d'analyse proposé par le LLM (optionnel).
        project_id: Identifiant unique du projet (auto-généré si None).

    Returns:
        AnalysisProject prêt à être sauvegardé.
    """
    if project_id is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        project_id = f"statforge_{timestamp}"

    # ── CORRECTION : Extraction robuste des infos (avec fallback) ──────────
    if plan is not None:
        proposed_method = getattr(plan, "proposed_method", result.method)
        outcome_variable = getattr(plan, "outcome_variable", "")
        predictor_variables = getattr(plan, "predictor_variables", [])
        reference_category = getattr(plan, "reference_category", None)
        study_design = getattr(plan, "study_design", "")
        confidence = getattr(plan, "confidence", "unknown")
        rationale = getattr(plan, "rationale", "")
        assumptions_to_check = getattr(plan, "assumptions_to_check", [])
    else:
        # Fallback : extraire les variables directement depuis result.variables
        proposed_method = result.method
        outcome_variable = ""
        predictor_variables = []
        reference_category = None
        study_design = "observational"
        confidence = "high"
        rationale = "Replay of saved analysis"
        assumptions_to_check = []

        for var_name, var_info in result.variables.items():
            role = var_info.get("role", "")
            name = var_info.get("name", var_name)
            if role == "outcome":
                outcome_variable = name
            elif role == "predictor":
                predictor_variables.append(name)

    # Sérialisation des diagnostics (objets → dicts)
    diagnostics_serialized = []
    for diag in result.diagnostics:
        diagnostics_serialized.append(
            {
                "code": diag.code,
                "severity": diag.severity.value,
                "message": diag.message,
                "value": diag.value if hasattr(diag, "value") else None,
            }
        )

    # Sérialisation des prétraitements
    preprocessing_serialized = []
    for prep in result.preprocessing:
        if hasattr(prep, "__dict__"):
            preprocessing_serialized.append(prep.__dict__)
        elif isinstance(prep, dict):
            preprocessing_serialized.append(prep)
        else:
            preprocessing_serialized.append({"description": str(prep)})

    return AnalysisProject(
        project_id=project_id,
        created_at=datetime.now(UTC).isoformat(),
        dataset_name=dataset_name,
        dataset_hash=dataset_hash,
        dataset_rows=dataset_rows,
        dataset_columns=dataset_columns,
        research_question=research_question,
        proposed_method=proposed_method,
        outcome_variable=outcome_variable,
        predictor_variables=predictor_variables,
        reference_category=reference_category,
        study_design=study_design,
        confidence=confidence,
        rationale=rationale,
        assumptions_to_check=assumptions_to_check,
        method=result.method,
        sample_size=result.sample_size,
        estimates=result.estimates,
        p_values=result.p_values,
        model_metrics=result.model_metrics,
        diagnostics=diagnostics_serialized,
        warnings=result.warnings,
        assumptions_checked=result.assumptions_checked,
        generated_code=result.generated_code,
        execution_log=result.execution_log,
        software_versions=result.software_versions,
        preprocessing=preprocessing_serialized,
        variables=result.variables,
    )


# ──────────────────────────────────────────────────────────────────────
# Sauvegarde d'un projet
# ──────────────────────────────────────────────────────────────────────


def save_analysis(
    project: AnalysisProject,
    output_dir: str = "projects",
) -> str:
    """Sauvegarde un projet d'analyse en JSON.

    Args:
        project: Projet à sauvegarder.
        output_dir: Répertoire de sortie (créé si inexistant).

    Returns:
        Chemin absolu vers le fichier JSON sauvegardé.

    Raises:
        IOError: Si l'écriture échoue.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"{project.project_id}.json"

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(project.to_dict(), f, indent=2, ensure_ascii=False)

        project.project_path = str(file_path)
        return str(file_path)

    except OSError as e:
        raise OSError(f"Impossible de sauvegarder le projet : {e}") from e


# ─────────────────────────────────────────────────────────────────────
# Chargement d'un projet
# ──────────────────────────────────────────────────────────────────────


def load_project(project_path: str) -> AnalysisProject:
    """Charge un projet d'analyse depuis un fichier JSON.

    Args:
        project_path: Chemin vers le fichier JSON.

    Returns:
        AnalysisProject chargé.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        json.JSONDecodeError: Si le fichier est corrompu.
    """
    path = Path(project_path)
    if not path.exists():
        raise FileNotFoundError(f"Projet introuvable : {project_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    project = AnalysisProject.from_dict(data)
    project.project_path = str(path)
    return project


# ─────────────────────────────────────────────────────────────────────
# Rejeu d'une analyse (T9 : via execute_tool)
# ──────────────────────────────────────────────────────────────────────


def replay_analysis(
    project: AnalysisProject,
    df: pd.DataFrame,
) -> AnalysisResult:
    """Rejoue une analyse sauvegardée via execute_tool (T9).

    Cette fonction ne fait PAS appel direct aux fonctions statistiques.
    Elle passe par le registre d'outils, garantissant la cohérence avec
    l'exécution originale.

    Args:
        project: Projet à rejouer.
        df: DataFrame contenant les données (doit correspondre au hash).

    Returns:
        AnalysisResult du rejeu.

    Raises:
        ValueError: Si les variables nécessaires ne sont pas disponibles.
    """
    # Normalisation du nom de la méthode
    method_name = project.proposed_method
    if not method_name.startswith("run_") and not method_name.startswith("check_"):
        method_name = f"run_{method_name}"

    # ── Préparation des arguments selon la méthode ──────────────────────
    if method_name == "run_chi_square":
        # CORRECTION : combiner outcome_variable + predictor_variables
        # car le LLM peut avoir mis la 1ère variable dans outcome_variable
        all_vars = []
        if project.outcome_variable:
            all_vars.append(project.outcome_variable)
        all_vars.extend(project.predictor_variables)

        if len(all_vars) < 2:
            raise ValueError(
                f"Le projet {project.project_id} n'a pas assez de variables "
                f"pour un chi_square (besoin de 2, trouvé {len(all_vars)}). "
                f"Variables disponibles : {all_vars}"
            )
        arguments = {
            "var1": all_vars[0],
            "var2": all_vars[1],
        }
    elif method_name in ("run_t_test", "run_anova"):
        if not project.outcome_variable:
            raise ValueError(
                f"Le projet {project.project_id} n'a pas d'outcome_variable "
                f"défini pour {method_name}."
            )
        if not project.predictor_variables:
            raise ValueError(
                f"Le projet {project.project_id} n'a pas de predictor_variables "
                f"défini pour {method_name}."
            )
        arguments = {
            "outcome": project.outcome_variable,
            "group": project.predictor_variables[0],
        }
    else:
        # Régression (linéaire, logistique, multinomiale)
        if not project.outcome_variable:
            raise ValueError(
                f"Le projet {project.project_id} n'a pas d'outcome_variable "
                f"défini pour {method_name}."
            )
        arguments = {
            "outcome": project.outcome_variable,
            "predictors": project.predictor_variables,
        }
        if project.reference_category:
            arguments["reference_category"] = project.reference_category

    # Exécution via execute_tool (T9)
    from app.agent.tools import execute_tool

    result = execute_tool(
        name=method_name,
        arguments=arguments,
        df=df,
    )

    return result


# ──────────────────────────────────────────────────────────────────────
# Liste des projets sauvegardés
# ──────────────────────────────────────────────────────────────────────


def list_projects(project_dir: str = "projects") -> list[dict[str, Any]]:
    """Liste tous les projets sauvegardés dans un répertoire.

    Args:
        project_dir: Répertoire contenant les projets.

    Returns:
        Liste de dictionnaires avec les métadonnées de chaque projet.
    """
    path = Path(project_dir)
    if not path.exists():
        return []

    projects = []
    for file_path in sorted(path.glob("*.json")):
        try:
            project = load_project(str(file_path))
            projects.append(
                {
                    "project_id": project.project_id,
                    "created_at": project.created_at,
                    "dataset_name": project.dataset_name,
                    "method": project.method,
                    "research_question": project.research_question,
                    "file_path": str(file_path),
                }
            )
        except Exception:
            # Ignore les fichiers corrompus
            continue

    return projects
