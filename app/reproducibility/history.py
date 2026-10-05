"""Journal chronologique append-only des analyses d'un projet.

Format : JSON Lines (un objet JSON par ligne), stocké dans
<project_dir>/history.jsonl.

Chaque entrée contient :
- project_id : identifiant unique de l'analyse
- timestamp : moment de la sauvegarde (ISO 8601)
- research_question : question de recherche
- method : méthode statistique utilisée
- dataset_name : nom du fichier source
- dataset_hash : hash SHA256 du dataset (intégrité)
- status : "success" | "failed" | "replayed"
- project_file : chemin vers le fichier JSON du projet
- replay_count : nombre de fois que l'analyse a été rejouée
- notes : commentaires optionnels

Contraintes respectées :
- T13 : traçabilité complète de chaque analyse
- T11 : simplicité (JSON Lines, pas de base de données)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────

HISTORY_FILENAME = "history.jsonl"


# ──────────────────────────────────────────────────────────────────────
# Fonctions publiques
# ──────────────────────────────────────────────────────────────────────

def append_to_history(
    project_dir: str,
    project_id: str,
    research_question: str,
    method: str,
    dataset_name: str,
    dataset_hash: str | None,
    project_file: str,
    status: str = "success",
    notes: str | None = None,
) -> str:
    """Ajoute une entrée au journal d'historique (append-only).

    Args:
        project_dir: Répertoire du projet (créé si inexistant).
        project_id: Identifiant unique de l'analyse.
        research_question: Question de recherche posée.
        method: Méthode statistique utilisée.
        dataset_name: Nom du fichier source.
        dataset_hash: Hash SHA256 du dataset (optionnel).
        project_file: Chemin vers le fichier JSON du projet.
        status: Statut de l'analyse ("success", "failed", "replayed").
        notes: Commentaires optionnels.

    Returns:
        Chemin vers le fichier history.jsonl.

    Raises:
        ValueError: Si le status n'est pas valide.
    """
    valid_statuses = {"success", "failed", "replayed"}
    if status not in valid_statuses:
        raise ValueError(
            f"Statut invalide : '{status}'. "
            f"Valeurs acceptées : {', '.join(sorted(valid_statuses))}"
        )

    project_path = Path(project_dir)
    project_path.mkdir(parents=True, exist_ok=True)

    history_file = project_path / HISTORY_FILENAME

    entry = {
        "project_id": project_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "research_question": research_question,
        "method": method,
        "dataset_name": dataset_name,
        "dataset_hash": dataset_hash,
        "status": status,
        "project_file": project_file,
        "replay_count": 0,
        "notes": notes,
    }

    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return str(history_file)


def get_history(project_dir: str) -> list[dict[str, Any]]:
    """Récupère toutes les entrées du journal d'historique.

    Args:
        project_dir: Répertoire du projet.

    Returns:
        Liste de dictionnaires, triée par timestamp (du plus récent au plus ancien).
        Liste vide si le fichier n'existe pas.
    """
    history_file = Path(project_dir) / HISTORY_FILENAME

    if not history_file.exists():
        return []

    entries = []
    with open(history_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    # Ignore les lignes corrompues
                    continue

    # Tri par timestamp décroissant (plus récent en premier)
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

    return entries


def get_analysis(project_dir: str, project_id: str) -> dict[str, Any] | None:
    """Récupère une entrée spécifique du journal par project_id.

    Args:
        project_dir: Répertoire du projet.
        project_id: Identifiant unique de l'analyse.

    Returns:
        Dictionnaire de l'entrée, ou None si non trouvée.
    """
    entries = get_history(project_dir)

    for entry in entries:
        if entry.get("project_id") == project_id:
            return entry

    return None


def update_replay_count(project_dir: str, project_id: str) -> None:
    """Incrémente le compteur de rejeu pour une analyse donnée.

    Note : cette fonction réécrit tout le fichier history.jsonl car
    le format JSON Lines ne supporte pas la modification in-place.

    Args:
        project_dir: Répertoire du projet.
        project_id: Identifiant de l'analyse à mettre à jour.

    Raises:
        ValueError: Si l'analyse n'est pas trouvée.
    """
    entries = get_history(project_dir)

    found = False
    for entry in entries:
        if entry.get("project_id") == project_id:
            entry["replay_count"] = entry.get("replay_count", 0) + 1
            entry["status"] = "replayed"
            found = True
            break

    if not found:
        raise ValueError(f"Analyse introuvable : {project_id}")

    # Réécriture du fichier (tri par timestamp original)
    history_file = Path(project_dir) / HISTORY_FILENAME
    with open(history_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_history_stats(project_dir: str) -> dict[str, Any]:
    """Calcule des statistiques sur le journal d'historique.

    Args:
        project_dir: Répertoire du projet.

    Returns:
        Dictionnaire avec :
        - total_analyses : nombre total d'analyses
        - by_method : dict méthode -> nombre
        - by_status : dict statut -> nombre
        - first_analysis : timestamp de la première analyse
        - last_analysis : timestamp de la dernière analyse
    """
    entries = get_history(project_dir)

    if not entries:
        return {
            "total_analyses": 0,
            "by_method": {},
            "by_status": {},
            "first_analysis": None,
            "last_analysis": None,
        }

    by_method: dict[str, int] = {}
    by_status: dict[str, int] = {}

    for entry in entries:
        method = entry.get("method", "unknown")
        status = entry.get("status", "unknown")

        by_method[method] = by_method.get(method, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1

    # Tri par timestamp croissant pour trouver first/last
    timestamps = [e.get("timestamp", "") for e in entries if e.get("timestamp")]
    timestamps.sort()

    return {
        "total_analyses": len(entries),
        "by_method": by_method,
        "by_status": by_status,
        "first_analysis": timestamps[0] if timestamps else None,
        "last_analysis": timestamps[-1] if timestamps else None,
    }
