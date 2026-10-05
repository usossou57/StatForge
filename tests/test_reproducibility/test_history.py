"""Tests pour le journal d'historique (T11.2)."""

import json
import tempfile
from pathlib import Path

import pytest

from app.reproducibility.history import (
    append_to_history,
    get_analysis,
    get_history,
    get_history_stats,
    update_replay_count,
)


@pytest.fixture
def project_dir(tmp_path):
    """Répertoire de projet temporaire."""
    return str(tmp_path / "test_project")


def test_append_to_history_creates_file(project_dir):
    """Vérifie que append_to_history crée le fichier history.jsonl."""
    history_file = append_to_history(
        project_dir=project_dir,
        project_id="test_001",
        research_question="Quel est l'effet de X sur Y ?",
        method="linear_regression",
        dataset_name="test.csv",
        dataset_hash="abc123",
        project_file="/path/to/test_001.json",
    )

    assert Path(history_file).exists()
    assert Path(history_file).name == "history.jsonl"


def test_append_to_history_adds_entry(project_dir):
    """Vérifie que append_to_history ajoute une entrée valide."""
    append_to_history(
        project_dir=project_dir,
        project_id="test_001",
        research_question="Question 1",
        method="linear_regression",
        dataset_name="test.csv",
        dataset_hash="abc123",
        project_file="/path/to/test_001.json",
    )

    entries = get_history(project_dir)
    assert len(entries) == 1
    assert entries[0]["project_id"] == "test_001"
    assert entries[0]["method"] == "linear_regression"
    assert entries[0]["status"] == "success"


def test_append_to_history_multiple_entries(project_dir):
    """Vérifie que plusieurs entrées peuvent être ajoutées."""
    for i in range(3):
        append_to_history(
            project_dir=project_dir,
            project_id=f"test_{i:03d}",
            research_question=f"Question {i}",
            method="linear_regression",
            dataset_name="test.csv",
            dataset_hash="abc123",
            project_file=f"/path/to/test_{i:03d}.json",
        )

    entries = get_history(project_dir)
    assert len(entries) == 3


def test_get_history_returns_empty_if_no_file(project_dir):
    """Vérifie que get_history retourne [] si le fichier n'existe pas."""
    entries = get_history(project_dir)
    assert entries == []


def test_get_history_sorted_by_timestamp(project_dir):
    """Vérifie que get_history trie par timestamp décroissant."""
    import time

    append_to_history(
        project_dir=project_dir,
        project_id="test_001",
        research_question="Question 1",
        method="linear_regression",
        dataset_name="test.csv",
        dataset_hash="abc123",
        project_file="/path/to/test_001.json",
    )

    time.sleep(0.01)  # Petite pause pour garantir un timestamp différent

    append_to_history(
        project_dir=project_dir,
        project_id="test_002",
        research_question="Question 2",
        method="chi_square",
        dataset_name="test.csv",
        dataset_hash="abc123",
        project_file="/path/to/test_002.json",
    )

    entries = get_history(project_dir)
    assert len(entries) == 2
    assert entries[0]["project_id"] == "test_002"  # Plus récent en premier
    assert entries[1]["project_id"] == "test_001"


def test_get_analysis_found(project_dir):
    """Vérifie que get_analysis trouve une entrée par project_id."""
    append_to_history(
        project_dir=project_dir,
        project_id="test_001",
        research_question="Question 1",
        method="linear_regression",
        dataset_name="test.csv",
        dataset_hash="abc123",
        project_file="/path/to/test_001.json",
    )

    entry = get_analysis(project_dir, "test_001")
    assert entry is not None
    assert entry["project_id"] == "test_001"


def test_get_analysis_not_found(project_dir):
    """Vérifie que get_analysis retourne None si non trouvé."""
    entry = get_analysis(project_dir, "nonexistent")
    assert entry is None


def test_update_replay_count(project_dir):
    """Vérifie que update_replay_count incrémente le compteur."""
    append_to_history(
        project_dir=project_dir,
        project_id="test_001",
        research_question="Question 1",
        method="linear_regression",
        dataset_name="test.csv",
        dataset_hash="abc123",
        project_file="/path/to/test_001.json",
    )

    update_replay_count(project_dir, "test_001")

    entry = get_analysis(project_dir, "test_001")
    assert entry["replay_count"] == 1
    assert entry["status"] == "replayed"


def test_update_replay_count_not_found(project_dir):
    """Vérifie que update_replay_count lève une erreur si non trouvé."""
    with pytest.raises(ValueError, match="Analyse introuvable"):
        update_replay_count(project_dir, "nonexistent")


def test_invalid_status(project_dir):
    """Vérifie que append_to_history rejette un status invalide."""
    with pytest.raises(ValueError, match="Statut invalide"):
        append_to_history(
            project_dir=project_dir,
            project_id="test_001",
            research_question="Question 1",
            method="linear_regression",
            dataset_name="test.csv",
            dataset_hash="abc123",
            project_file="/path/to/test_001.json",
            status="invalid_status",
        )


def test_get_history_stats(project_dir):
    """Vérifie que get_history_stats calcule les statistiques."""
    append_to_history(
        project_dir=project_dir,
        project_id="test_001",
        research_question="Question 1",
        method="linear_regression",
        dataset_name="test.csv",
        dataset_hash="abc123",
        project_file="/path/to/test_001.json",
    )

    append_to_history(
        project_dir=project_dir,
        project_id="test_002",
        research_question="Question 2",
        method="chi_square",
        dataset_name="test.csv",
        dataset_hash="abc123",
        project_file="/path/to/test_002.json",
    )

    stats = get_history_stats(project_dir)
    assert stats["total_analyses"] == 2
    assert stats["by_method"]["linear_regression"] == 1
    assert stats["by_method"]["chi_square"] == 1
    assert stats["by_status"]["success"] == 2


def test_get_history_stats_empty(project_dir):
    """Vérifie que get_history_stats gère le cas vide."""
    stats = get_history_stats(project_dir)
    assert stats["total_analyses"] == 0
    assert stats["by_method"] == {}
