"""Tests pour les outils de dictionnaire de données enrichi."""

from __future__ import annotations

import copy

import pytest

from app.data.dictionary_tools import set_variable_metadata
from app.data.models import DataDictionary, Variable


@pytest.fixture
def sample_dd() -> DataDictionary:
    """Retourne un DataDictionary de test."""
    return DataDictionary(
        dataset_name="test.csv",
        n_rows=1000,
        n_columns=2,
        variables={
            "age": Variable(
                name="age",
                dtype="numeric",
                semantic_type="ratio",
            ),
            "revenu": Variable(
                name="revenu",
                dtype="numeric",
                semantic_type="ratio",
                metadata={"source": "enquête"},
            ),
        },
        source_path="/tmp/test.csv",
        file_hash="abc123",
    )


# ============================================================================
# Tests pour set_variable_metadata
# ============================================================================


def test_set_variable_metadata_cas_nominal(sample_dd: DataDictionary) -> None:
    """Test de mise à jour complète des métadonnées."""
    dd_updated = set_variable_metadata(
        sample_dd,
        "age",
        unit="années",
        measure_level="ratio",
        role="predictor",
        weight="poids_age",
    )

    # Vérifier que l'original n'est pas modifié (immuabilité)
    assert sample_dd.variables["age"].metadata == {}
    assert sample_dd.variables["age"].role is None
    assert sample_dd.variables["age"].weight is None

    # Vérifier les mises à jour
    updated_age = dd_updated.variables["age"]
    assert updated_age.metadata == {"unit": "années"}
    assert updated_age.semantic_type == "ratio"
    assert updated_age.role == "predictor"
    assert updated_age.weight == "poids_age"

    # Vérifier que les autres champs sont préservés
    assert updated_age.dtype == "numeric"
    assert updated_age.name == "age"


def test_set_variable_metadata_mise_a_jour_partielle(sample_dd: DataDictionary) -> None:
    """Test de mise à jour partielle (fusion avec metadata existant)."""
    dd_updated = set_variable_metadata(
        sample_dd,
        "revenu",
        unit="euros",
        role="outcome",
    )

    updated_revenu = dd_updated.variables["revenu"]
    # Le metadata existant doit être préservé et fusionné
    assert updated_revenu.metadata == {"source": "enquête", "unit": "euros"}
    assert updated_revenu.role == "outcome"
    # Le weight n'était pas défini, il reste None
    assert updated_revenu.weight is None
    # Le semantic_type n'était pas modifié, il reste "ratio"
    assert updated_revenu.semantic_type == "ratio"


def test_set_variable_metadata_variable_inexistante(sample_dd: DataDictionary) -> None:
    """Test de mise à jour d'une variable inexistante."""
    with pytest.raises(KeyError, match="n'existe pas"):
        set_variable_metadata(sample_dd, "inexistante", unit="kg")


def test_set_variable_metadata_niveau_mesure_invalide(sample_dd: DataDictionary) -> None:
    """Test avec un niveau de mesure invalide."""
    with pytest.raises(ValueError, match="Niveau de mesure invalide"):
        set_variable_metadata(sample_dd, "age", measure_level="continu")  # type: ignore


def test_set_variable_metadata_role_invalide(sample_dd: DataDictionary) -> None:
    """Test avec un rôle de variable invalide."""
    with pytest.raises(ValueError, match="Rôle de variable invalide"):
        set_variable_metadata(sample_dd, "age", role="cible")  # type: ignore


def test_set_variable_metadata_immuabilite(sample_dd: DataDictionary) -> None:
    """Vérifie que la fonction ne modifie pas le dictionnaire d'entrée."""
    dd_original = copy.deepcopy(sample_dd)
    _dd_updated = set_variable_metadata(sample_dd, "age", role="predictor")

    assert sample_dd == dd_original