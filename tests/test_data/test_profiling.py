"""Tests pour le module de profilage de données."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from app.data.models import DataDictionary
from app.data.profiling import build_data_dictionary, describe_variable, profile_dataset


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame simple pour les tests."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "sexe": ["H", "F", "H", "F", "H"],
            "score": [85.5, 92.0, 78.3, 88.7, 95.2],
        }
    )


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> str:
    """Crée un fichier CSV de test et retourne son chemin."""
    df = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "sexe": ["H", "F", "H", "F", "H"],
            "score": [85.5, 92.0, 78.3, 88.7, 95.2],
        }
    )
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    """DataFrame avec des cas limites (constante, manquants, catégorie rare)."""
    return pd.DataFrame(
        {
            "constante": [1, 1, 1, 1, 1],
            "age": [25.0, 30.0, 35.0, 40.0, 45.0],
            "revenu": [1000.0, 2000.0, 3000.0, None, None],
            "categorie_rare": ["A", "A", "A", "A", "B"],
            "binaire": [0, 0, 0, 1, 0],
        }
    )


def test_profile_dataset_cas_nominal(sample_df: pd.DataFrame) -> None:
    profile = profile_dataset(sample_df)
    assert profile["n_rows"] == 5
    assert profile["n_columns"] == 3
    assert "age" in profile["dtypes"]
    assert profile["missing_pct"]["age"] == 0.0
    assert profile["n_unique"]["sexe"] == 2
    assert "constante" not in profile["constant_columns"]


def test_profile_dataset_colonne_constante() -> None:
    df = pd.DataFrame({"id": [1, 1, 1, 1], "val": [10, 20, 30, 40]})
    profile = profile_dataset(df)
    assert "id" in profile["constant_columns"]
    assert "val" not in profile["constant_columns"]


def test_profile_dataset_valeurs_manquantes() -> None:
    df = pd.DataFrame({"a": [1, 2, None, None, None]})
    profile = profile_dataset(df)
    assert profile["missing_pct"]["a"] == pytest.approx(60.0)


def test_profile_dataset_categorie_rare() -> None:
    df = pd.DataFrame({"cat": ["A", "A", "A", "A", "B"]})
    profile = profile_dataset(df)
    assert profile["n_unique"]["cat"] == 2


def test_describe_variable_numerique(sample_df: pd.DataFrame) -> None:
    desc = describe_variable(sample_df, "age")
    assert "mean" in desc
    assert desc["mean"] == 35.0
    assert desc["n_missing"] == 0


def test_describe_variable_categorielle(sample_df: pd.DataFrame) -> None:
    desc = describe_variable(sample_df, "sexe")
    assert "frequencies" in desc
    assert desc["frequencies"]["H"] == 3
    assert desc["frequencies"]["F"] == 2


def test_describe_variable_colonne_inexistante(sample_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        describe_variable(sample_df, "inexistant")


def test_describe_variable_avec_manquants() -> None:
    df = pd.DataFrame({"val": [10.0, 20.0, None, None]})
    desc = describe_variable(df, "val")
    assert desc["n_missing"] == 2
    assert desc["mean"] == 15.0


def test_build_data_dictionary_cas_nominal(sample_df: pd.DataFrame, sample_csv_path: str) -> None:
    data_dict = build_data_dictionary(sample_df, "test_dataset", sample_csv_path)
    assert isinstance(data_dict, DataDictionary)
    assert data_dict.dataset_name == "test_dataset"
    assert data_dict.n_rows == 5
    assert data_dict.n_columns == 3
    assert len(data_dict.file_hash) == 64

    # CORRECTION : age a 5 valeurs uniques entières -> "count" (pas "continuous" car <= 10)
    age_var = data_dict.variables["age"]
    assert age_var.dtype == "numeric"
    assert age_var.semantic_type == "count"


def test_build_data_dictionary_hash_calcule(sample_df: pd.DataFrame, sample_csv_path: str) -> None:
    data_dict1 = build_data_dictionary(sample_df, "test", sample_csv_path)
    data_dict2 = build_data_dictionary(sample_df, "test", sample_csv_path)
    assert data_dict1.file_hash == data_dict2.file_hash


def test_build_data_dictionary_typage_complet(dirty_df: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        dirty_df.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        data_dict = build_data_dictionary(dirty_df, "dirty_dataset", temp_path)

        # CORRECTION : 1 valeur unique entière -> "count" (pas "binary" qui nécessite exactement 2)
        const_var = data_dict.variables["constante"]
        assert const_var.dtype == "numeric"
        assert const_var.semantic_type == "count"

        bin_var = data_dict.variables["binaire"]
        assert bin_var.dtype == "numeric"
        assert bin_var.semantic_type == "binary"

        cat_var = data_dict.variables["categorie_rare"]
        assert cat_var.dtype == "categorical"
        assert cat_var.semantic_type == "binary"
    finally:
        os.remove(temp_path)


def test_build_data_dictionary_fichier_inexistant(sample_df: pd.DataFrame) -> None:
    with pytest.raises(FileNotFoundError):
        build_data_dictionary(sample_df, "test", "/chemin/inexistant.csv")


def test_build_data_dictionary_semantic_types() -> None:
    # CORRECTION : Toutes les listes ont maintenant exactement 11 éléments
    df = pd.DataFrame(
        {
            "binaire": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "count": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1],
            "continuous": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1],
        }
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        data_dict = build_data_dictionary(df, "semantic_test", temp_path)
        assert data_dict.variables["binaire"].semantic_type == "binary"
        assert data_dict.variables["count"].semantic_type == "count"
        assert data_dict.variables["continuous"].semantic_type == "continuous"
    finally:
        os.remove(temp_path)
