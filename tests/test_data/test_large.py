"""Tests pour le module de gestion des gros volumes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.large import (
    downcast_dtypes,
    import_csv_chunked,
    memory_report,
    sample_for_analysis,
)
from app.errors import InvalidMethodInputError


@pytest.fixture
def large_df() -> pd.DataFrame:
    """Retourne un DataFrame de 10 000 lignes pour les tests."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "id": range(10_000),
            "age": np.random.randint(18, 90, size=10_000),
            "score": np.random.uniform(0, 100, size=10_000),
            "category": np.random.choice(["A", "B", "C", "D"], size=10_000),
            "small_int": np.random.randint(0, 100, size=10_000),
        }
    )


# ============================================================================
# Tests pour import_csv_chunked
# ============================================================================


def test_import_csv_chunked_cas_nominal(tmp_path) -> None:
    """Test d'import chunké avec un CSV valide."""
    # Créer un CSV de test
    df_orig = pd.DataFrame(
        {
            "id": range(1000),
            "value": np.random.uniform(0, 100, size=1000),
        }
    )
    csv_path = tmp_path / "test.csv"
    df_orig.to_csv(csv_path, index=False)

    # Importer par morceaux
    df_imported = import_csv_chunked(str(csv_path), chunksize=300)

    assert len(df_imported) == 1000
    assert list(df_imported.columns) == ["id", "value"]
    pd.testing.assert_frame_equal(df_imported, df_orig)


def test_import_csv_chunked_usecols(tmp_path) -> None:
    """Test d'import chunké avec sélection de colonnes."""
    df_orig = pd.DataFrame(
        {
            "id": range(500),
            "col1": np.random.uniform(0, 100, size=500),
            "col2": np.random.uniform(0, 100, size=500),
            "col3": np.random.uniform(0, 100, size=500),
        }
    )
    csv_path = tmp_path / "test.csv"
    df_orig.to_csv(csv_path, index=False)

    # Importer seulement id et col1
    df_imported = import_csv_chunked(str(csv_path), chunksize=200, usecols=["id", "col1"])

    assert len(df_imported) == 500
    assert list(df_imported.columns) == ["id", "col1"]


def test_import_csv_chunked_fichier_inexistant() -> None:
    """Test d'import chunké avec un fichier inexistant."""
    with pytest.raises(FileNotFoundError, match="n'existe pas"):
        import_csv_chunked("/chemin/inexistant/fichier.csv")


def test_import_csv_chunked_fichier_vide(tmp_path) -> None:
    """Test d'import chunké avec un fichier vide."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("col1,col2\n")

    df_imported = import_csv_chunked(str(csv_path), chunksize=100)
    assert len(df_imported) == 0


# ============================================================================
# Tests pour downcast_dtypes
# ============================================================================


def test_downcast_dtypes_int64_to_int8(large_df: pd.DataFrame) -> None:
    """Test de conversion int64 → int8 pour petites valeurs."""
    df_orig = large_df.copy()
    df_opt = downcast_dtypes(large_df)

    # small_int (0-99) devrait être converti en int8
    assert df_opt["small_int"].dtype == np.int8
    # age (18-89) devrait être converti en int8 aussi
    assert df_opt["age"].dtype == np.int8
    # id (0-9999) devrait être converti en int16 ou int32
    assert df_opt["id"].dtype in [np.int16, np.int32]

    # Vérifier que l'original n'est pas modifié
    assert large_df.equals(df_orig)


def test_downcast_dtypes_float64_to_float32(large_df: pd.DataFrame) -> None:
    """Test de conversion float64 → float32."""
    df_opt = downcast_dtypes(large_df)

    # score (0-100) devrait être converti en float32
    assert df_opt["score"].dtype == np.float32


def test_downcast_dtypes_object_to_category(large_df: pd.DataFrame) -> None:
    """Test de conversion object → category."""
    df_opt = downcast_dtypes(large_df)

    # category (4 valeurs uniques sur 10 000 lignes) devrait être category
    assert df_opt["category"].dtype.name == "category"


def test_downcast_dtypes_non_mutation(large_df: pd.DataFrame) -> None:
    """Vérifie que downcast_dtypes ne modifie pas l'entrée."""
    df_orig = large_df.copy()
    _df_opt = downcast_dtypes(large_df)
    assert large_df.equals(df_orig)


# ============================================================================
# Tests pour memory_report
# ============================================================================


def test_memory_report_cas_nominal(large_df: pd.DataFrame) -> None:
    """Test de rapport mémoire."""
    report = memory_report(large_df)

    assert "total_memory_mb" in report
    assert "total_memory_gb" in report
    assert "n_rows" in report
    assert "n_columns" in report
    assert "by_column" in report
    assert "dtypes_summary" in report

    assert report["n_rows"] == 10_000
    assert report["n_columns"] == 5
    assert report["total_memory_mb"] > 0
    assert "id" in report["by_column"]
    assert "dtype" in report["by_column"]["id"]
    assert "memory_mb" in report["by_column"]["id"]


def test_memory_report_json_serializable(large_df: pd.DataFrame) -> None:
    """Vérifie que le rapport est JSON-sérialisable (T23)."""
    import json

    report = memory_report(large_df)

    # Ne doit pas lever d'exception
    json_str = json.dumps(report)
    assert isinstance(json_str, str)

    # Recharger et vérifier
    report_loaded = json.loads(json_str)
    assert report_loaded["n_rows"] == 10_000


def test_memory_report_dataframe_vide() -> None:
    """Test de rapport mémoire sur un DataFrame vide."""
    df_empty = pd.DataFrame()
    report = memory_report(df_empty)

    assert report["n_rows"] == 0
    assert report["n_columns"] == 0
    assert report["total_memory_mb"] == pytest.approx(0.0, abs=0.01)


# ============================================================================
# Tests pour sample_for_analysis
# ============================================================================


def test_sample_for_analysis_pas_d_echantillonnage(large_df: pd.DataFrame) -> None:
    """Test quand le DataFrame est déjà plus petit que max_rows."""
    df_sample, stats = sample_for_analysis(large_df, max_rows=20_000)

    assert len(df_sample) == 10_000
    assert stats["was_sampled"] is False
    assert stats["sampling_rate"] == 1.0
    assert stats["original_rows"] == 10_000
    assert stats["sampled_rows"] == 10_000


def test_sample_for_analysis_echantillonnage(large_df: pd.DataFrame) -> None:
    """Test d'échantillonnage quand le DataFrame dépasse max_rows."""
    df_sample, stats = sample_for_analysis(large_df, max_rows=5_000)

    assert len(df_sample) == 5_000
    assert stats["was_sampled"] is True
    assert stats["sampling_rate"] == pytest.approx(0.5, rel=1e-3)
    assert stats["original_rows"] == 10_000
    assert stats["sampled_rows"] == 5_000
    assert stats["random_state"] == 0


def test_sample_for_analysis_reproductible(large_df: pd.DataFrame) -> None:
    """Vérifie que la même graine donne le même échantillon."""
    df_s1, _ = sample_for_analysis(large_df, max_rows=3_000, random_state=42)
    df_s2, _ = sample_for_analysis(large_df, max_rows=3_000, random_state=42)

    pd.testing.assert_frame_equal(df_s1, df_s2)


def test_sample_for_analysis_graine_differente(large_df: pd.DataFrame) -> None:
    """Vérifie que des graines différentes donnent des échantillons différents."""
    df_s1, _ = sample_for_analysis(large_df, max_rows=3_000, random_state=0)
    df_s2, _ = sample_for_analysis(large_df, max_rows=3_000, random_state=1)

    # Les DataFrames ne doivent pas être identiques
    assert not df_s1.equals(df_s2)


def test_sample_for_analysis_max_rows_invalide(large_df: pd.DataFrame) -> None:
    """Test avec max_rows <= 0."""
    with pytest.raises(InvalidMethodInputError, match="strictement positif"):
        sample_for_analysis(large_df, max_rows=0)

    with pytest.raises(InvalidMethodInputError, match="strictement positif"):
        sample_for_analysis(large_df, max_rows=-100)


def test_sample_for_analysis_non_mutation(large_df: pd.DataFrame) -> None:
    """Vérifie que sample_for_analysis ne modifie pas l'entrée."""
    df_orig = large_df.copy()
    _df_sample, _stats = sample_for_analysis(large_df, max_rows=5_000)
    assert large_df.equals(df_orig)


# ============================================================================
# Test de performance (marqué slow)
# ============================================================================


@pytest.mark.slow
def test_import_csv_chunked_performance(tmp_path) -> None:
    """Test de performance sur un CSV de 500 000 lignes.

    Ce test vérifie que l'import chunké reste sous 4 Go de RAM (T22).
    """
    # Créer un CSV de 500 000 lignes
    np.random.seed(42)
    df_large = pd.DataFrame(
        {
            "id": range(500_000),
            "value1": np.random.uniform(0, 100, size=500_000),
            "value2": np.random.uniform(0, 100, size=500_000),
            "category": np.random.choice(["A", "B", "C", "D", "E"], size=500_000),
        }
    )
    csv_path = tmp_path / "large.csv"
    df_large.to_csv(csv_path, index=False)

    # Importer par morceaux
    df_imported = import_csv_chunked(str(csv_path), chunksize=100_000)

    assert len(df_imported) == 500_000
    assert list(df_imported.columns) == ["id", "value1", "value2", "category"]

    # Vérifier la mémoire (devrait être < 4 Go)
    report = memory_report(df_imported)
    assert report["total_memory_gb"] < 4.0
