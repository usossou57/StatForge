"""Tests pour les modèles de données (Variable et DataDictionary).

Ce module teste la sérialisation et la désérialisation des objets
Variable et DataDictionary via les méthodes to_json() et from_json().
"""

import json

from app.data.models import DataDictionary, Variable


def test_variable_creation_minimale() -> None:
    """Test de création d'une Variable avec uniquement le nom."""
    var = Variable(name="age")
    assert var.name == "age"
    assert var.label is None
    assert var.dtype == "unknown"
    assert var.semantic_type is None
    assert var.levels is None
    assert var.labels is None
    assert var.missing_values == []
    assert var.role is None
    assert var.reference_category is None
    assert var.weight is None
    assert var.metadata == {}


def test_variable_creation_complete() -> None:
    """Test de création d'une Variable avec tous les champs renseignés."""
    var = Variable(
        name="sexe",
        label="Sexe du répondant",
        dtype="categorical",
        semantic_type="binary",
        levels=["homme", "femme"],
        labels={"0": "homme", "1": "femme"},
        missing_values=[99, -1],
        role="predictor",
        reference_category="homme",
        weight=None,
        metadata={"source": "questionnaire", "question_number": 3},
    )
    assert var.name == "sexe"
    assert var.label == "Sexe du répondant"
    assert var.dtype == "categorical"
    assert var.semantic_type == "binary"
    assert var.levels == ["homme", "femme"]
    assert var.labels == {"0": "homme", "1": "femme"}
    assert var.missing_values == [99, -1]
    assert var.role == "predictor"
    assert var.reference_category == "homme"
    assert var.weight is None
    assert var.metadata == {"source": "questionnaire", "question_number": 3}


def test_data_dictionary_to_json_produit_json_valide() -> None:
    """Test que to_json() produit une chaîne JSON valide et parsable."""
    data_dict = DataDictionary(
        dataset_name="test_dataset",
        n_rows=100,
        n_columns=3,
        variables={
            "age": Variable(name="age", dtype="numeric"),
            "sexe": Variable(name="sexe", dtype="categorical"),
        },
        source_path="/data/test.csv",
        file_hash="abc123",
    )
    json_str = data_dict.to_json()

    # Vérifier que c'est une chaîne
    assert isinstance(json_str, str)

    # Vérifier que c'est du JSON valide
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)
    assert parsed["dataset_name"] == "test_dataset"
    assert parsed["n_rows"] == 100
    assert parsed["n_columns"] == 3


def test_data_dictionary_roundtrip_complet() -> None:
    """Test de sérialisation/désérialisation avec plusieurs variables complètes."""
    original = DataDictionary(
        dataset_name="enquete_sante",
        n_rows=500,
        n_columns=4,
        variables={
            "age": Variable(
                name="age",
                label="Âge en années",
                dtype="numeric",
                semantic_type="continuous",
                role="predictor",
            ),
            "sexe": Variable(
                name="sexe",
                label="Sexe du répondant",
                dtype="categorical",
                semantic_type="binary",
                levels=["homme", "femme"],
                labels={"0": "homme", "1": "femme"},
                role="predictor",
                reference_category="homme",
            ),
            "revenu": Variable(
                name="revenu",
                label="Revenu mensuel",
                dtype="numeric",
                semantic_type="continuous",
                missing_values=[-999, -1],
                role="outcome",
                metadata={"unite": "euros", "periode": "mensuel"},
            ),
            "poids": Variable(
                name="poids",
                label="Poids sondage",
                dtype="numeric",
                role="weight",
            ),
        },
        source_path="/data/enquete_sante.sav",
        file_hash="def456ghi789",
    )

    json_str = original.to_json()
    reconstructed = DataDictionary.from_json(json_str)

    # Vérifier l'égalité des champs primitifs
    assert reconstructed.dataset_name == original.dataset_name
    assert reconstructed.n_rows == original.n_rows
    assert reconstructed.n_columns == original.n_columns
    assert reconstructed.source_path == original.source_path
    assert reconstructed.file_hash == original.file_hash

    # Vérifier l'égalité des variables
    assert len(reconstructed.variables) == len(original.variables)
    for var_name in original.variables:
        assert var_name in reconstructed.variables
        orig_var = original.variables[var_name]
        recon_var = reconstructed.variables[var_name]
        assert recon_var.name == orig_var.name
        assert recon_var.label == orig_var.label
        assert recon_var.dtype == orig_var.dtype
        assert recon_var.semantic_type == orig_var.semantic_type
        assert recon_var.levels == orig_var.levels
        assert recon_var.labels == orig_var.labels
        assert recon_var.missing_values == orig_var.missing_values
        assert recon_var.role == orig_var.role
        assert recon_var.reference_category == orig_var.reference_category
        assert recon_var.weight == orig_var.weight
        assert recon_var.metadata == orig_var.metadata


def test_data_dictionary_champs_optionnels_vides() -> None:
    """Test avec des Variable ayant tous les champs optionnels à None ou vides."""
    original = DataDictionary(
        dataset_name="minimal_dataset",
        n_rows=10,
        n_columns=2,
        variables={
            "var1": Variable(name="var1"),
            "var2": Variable(
                name="var2",
                label=None,
                dtype="unknown",
                semantic_type=None,
                levels=None,
                labels=None,
                missing_values=[],
                role=None,
                reference_category=None,
                weight=None,
                metadata={},
            ),
        },
        source_path="/data/minimal.csv",
        file_hash="xyz789",
    )

    json_str = original.to_json()
    reconstructed = DataDictionary.from_json(json_str)

    assert reconstructed.dataset_name == original.dataset_name
    assert reconstructed.n_rows == original.n_rows
    assert reconstructed.n_columns == original.n_columns
    assert reconstructed.source_path == original.source_path
    assert reconstructed.file_hash == original.file_hash

    for var_name in original.variables:
        orig_var = original.variables[var_name]
        recon_var = reconstructed.variables[var_name]
        assert recon_var.name == orig_var.name
        assert recon_var.label is None
        assert recon_var.semantic_type is None
        assert recon_var.levels is None
        assert recon_var.labels is None
        assert recon_var.missing_values == []
        assert recon_var.role is None
        assert recon_var.reference_category is None
        assert recon_var.weight is None
        assert recon_var.metadata == {}


def test_from_json_reconstruit_objet_identique() -> None:
    """Test que from_json() reconstruit un objet structurellement identique."""
    original = DataDictionary(
        dataset_name="test_identique",
        n_rows=42,
        n_columns=1,
        variables={
            "test_var": Variable(
                name="test_var",
                label="Variable de test",
                dtype="categorical",
                semantic_type="ordinal",
                levels=["bas", "moyen", "haut"],
                labels={"1": "bas", "2": "moyen", "3": "haut"},
                missing_values=[0, -1, 999],
                role="predictor",
                reference_category="bas",
                weight="poids_var",
                metadata={"cle1": "valeur1", "cle2": 123},
            )
        },
        source_path="/chemin/vers/fichier.sav",
        file_hash="hash_de_test_12345",
    )

    json_str = original.to_json()
    reconstructed = DataDictionary.from_json(json_str)

    orig_var = original.variables["test_var"]
    recon_var = reconstructed.variables["test_var"]

    assert recon_var.name == orig_var.name
    assert recon_var.label == orig_var.label
    assert recon_var.dtype == orig_var.dtype
    assert recon_var.semantic_type == orig_var.semantic_type
    assert recon_var.levels == orig_var.levels
    assert recon_var.labels == orig_var.labels
    assert recon_var.missing_values == orig_var.missing_values
    assert recon_var.role == orig_var.role
    assert recon_var.reference_category == orig_var.reference_category
    assert recon_var.weight == orig_var.weight
    assert recon_var.metadata == orig_var.metadata


def test_to_json_gere_caracteres_speciaux() -> None:
    """Test que to_json() gère correctement les caractères spéciaux (UTF-8)."""
    original = DataDictionary(
        dataset_name="données_françaises",
        n_rows=100,
        n_columns=1,
        variables={
            "réponse": Variable(
                name="réponse",
                label="Réponse à la question",
                dtype="text",
                metadata={"clé": "valeur_avec_accents_éèàù"},
            )
        },
        source_path="/données/fichier.csv",
        file_hash="hash_avec_émojis_🎉",
    )

    json_str = original.to_json()
    reconstructed = DataDictionary.from_json(json_str)

    assert reconstructed.dataset_name == "données_françaises"
    assert reconstructed.variables["réponse"].label == "Réponse à la question"
    assert reconstructed.variables["réponse"].metadata["clé"] == "valeur_avec_accents_éèàù"
    assert reconstructed.source_path == "/données/fichier.csv"
    assert reconstructed.file_hash == "hash_avec_émojis_🎉"
