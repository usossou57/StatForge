"""Outils pour le dictionnaire de données enrichi (V2 T15.10 / T17).

Ce module fournit des fonctions pour mettre à jour les métadonnées des
variables dans un DataDictionary existant, sans ajouter de nouveaux champs
au modèle de données (respect strict du contrat T17).
"""

from __future__ import annotations

from typing import Literal

from app.data.models import DataDictionary, Variable

# Niveaux de mesure statistiques valides (mappés sur semantic_type)
MeasureLevel = Literal["nominal", "ordinal", "interval", "ratio"]

# Rôles de variables typiques dans les analyses
VariableRole = Literal["outcome", "predictor", "weight", "id", "strata", "other"]


def set_variable_metadata(
    dd: DataDictionary,
    name: str,
    *,
    unit: str | None = None,
    measure_level: MeasureLevel | None = None,
    role: VariableRole | None = None,
    weight: str | None = None,
) -> DataDictionary:
    """Met à jour les métadonnées d'une variable dans un dictionnaire de données.

    Cette fonction ne modifie pas le dictionnaire d'entrée (immuabilité).
    Elle retourne une nouvelle instance de DataDictionary avec la variable
    mise à jour.

    Args:
        dd: Dictionnaire de données d'entrée (non modifié).
        name: Nom de la variable à mettre à jour.
        unit: Unité de mesure (ex: "kg", "années", "euros"). Sera stockée
            dans Variable.metadata["unit"].
        measure_level: Niveau de mesure statistique. Sera stocké dans
            Variable.semantic_type. Doit être l'un des suivants :
            "nominal", "ordinal", "interval", "ratio".
        role: Rôle de la variable dans les analyses.
            Ex: "outcome", "predictor", "weight", "id", "strata", "other".
        weight: Nom de la variable de pondération associée.

    Returns:
        Un nouveau DataDictionary avec la variable mise à jour.

    Raises:
        KeyError: Si la variable n'existe pas dans le dictionnaire.
        ValueError: Si le niveau de mesure ou le rôle fourni est invalide.
    """
    if name not in dd.variables:
        raise KeyError(f"La variable '{name}' n'existe pas dans le dictionnaire.")

    valid_measure_levels = {"nominal", "ordinal", "interval", "ratio"}
    if measure_level is not None and measure_level not in valid_measure_levels:
        raise ValueError(
            f"Niveau de mesure invalide : '{measure_level}'. "
            f"Valeurs autorisées : {', '.join(sorted(valid_measure_levels))}."
        )

    valid_roles = {"outcome", "predictor", "weight", "id", "strata", "other"}
    if role is not None and role not in valid_roles:
        raise ValueError(
            f"Rôle de variable invalide : '{role}'. "
            f"Valeurs autorisées : {', '.join(sorted(valid_roles))}."
        )

    original_var = dd.variables[name]

    # Mise à jour des métadonnées (fusion avec l'existant)
    new_metadata = dict(original_var.metadata or {})
    if unit is not None:
        new_metadata["unit"] = unit

    # Création de la nouvelle variable en préservant TOUS les champs existants (T17)
    new_var = Variable(
        name=original_var.name,
        label=original_var.label,
        dtype=original_var.dtype,
        semantic_type=measure_level
        if measure_level is not None
        else original_var.semantic_type,
        levels=original_var.levels,
        labels=original_var.labels,
        missing_values=list(original_var.missing_values),
        role=role if role is not None else original_var.role,
        reference_category=original_var.reference_category,
        weight=weight if weight is not None else original_var.weight,
        metadata=new_metadata,
    )

    # Création d'un nouveau dictionnaire avec la variable mise à jour
    new_variables = dict(dd.variables)
    new_variables[name] = new_var

    return DataDictionary(
        dataset_name=dd.dataset_name,
        n_rows=dd.n_rows,
        n_columns=dd.n_columns,
        variables=new_variables,
        source_path=dd.source_path,
        file_hash=dd.file_hash,
    )
