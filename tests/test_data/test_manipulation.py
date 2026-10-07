"""Tests pour le module de manipulation de DataFrames."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data.manipulation import (
    aggregate_by,
    drop_duplicates,
    merge_datasets,
    pivot_longer,
    pivot_wider,
    sample_rows,
    transpose_dataset,
)
from app.errors import InvalidMethodInputError


@pytest.fixture
def df_left() -> pd.DataFrame:
    """DataFrame de gauche pour les tests de fusion."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "nom": ["Alice", "Bob", "Charlie", "Diana"],
            "age": [25, 30, 35, 40],
        }
    )


@pytest.fixture
def df_right() -> pd.DataFrame:
    """DataFrame de droite pour les tests de fusion."""
    return pd.DataFrame(
        {
            "id": [2, 3, 4, 5],
            "score": [85.5, 92.0, 78.3, 95.2],
            "groupe": ["A", "B", "A", "B"],
        }
    )


@pytest.fixture
def df_wide() -> pd.DataFrame:
    """DataFrame au format large pour les tests de pivot."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "score_t1": [80, 90, 70],
            "score_t2": [85, 88, 75],
            "score_t3": [90, 92, 80],
        }
    )


@pytest.fixture
def df_long() -> pd.DataFrame:
    """DataFrame au format long pour les tests de pivot."""
    return pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "temps": ["t1", "t2", "t1", "t2", "t1", "t2"],
            "score": [80, 85, 90, 88, 70, 75],
        }
    )


# ============================================================================
# Tests pour merge_datasets
# ============================================================================


def test_merge_inner(df_left: pd.DataFrame, df_right: pd.DataFrame) -> None:
    """Test de fusion interne."""
    df_merged, step = merge_datasets(df_left, df_right, on="id", how="inner")
    assert len(df_merged) == 3  # ids 2, 3, 4
    assert "nom" in df_merged.columns
    assert "score" in df_merged.columns
    assert step.operation == "merge"
    assert step.params["how"] == "inner"


def test_merge_left(df_left: pd.DataFrame, df_right: pd.DataFrame) -> None:
    """Test de fusion gauche."""
    df_merged, _step = merge_datasets(df_left, df_right, on="id", how="left")
    assert len(df_merged) == 4  # tous les ids de gauche


def test_merge_right(df_left: pd.DataFrame, df_right: pd.DataFrame) -> None:
    """Test de fusion droite."""
    df_merged, _step = merge_datasets(df_left, df_right, on="id", how="right")
    assert len(df_merged) == 4  # tous les ids de droite


def test_merge_outer(df_left: pd.DataFrame, df_right: pd.DataFrame) -> None:
    """Test de fusion externe."""
    df_merged, _step = merge_datasets(df_left, df_right, on="id", how="outer")
    assert len(df_merged) == 5  # ids 1 à 5


def test_merge_cles_dupliquees() -> None:
    """Test de fusion avec clés dupliquées (avertissement)."""
    df_l = pd.DataFrame({"id": [1, 1, 2], "val_l": ["a", "b", "c"]})
    df_r = pd.DataFrame({"id": [1, 2], "val_r": ["x", "y"]})
    df_merged, step = merge_datasets(df_l, df_r, on="id", how="inner")
    # 2 lignes pour id=1 (produit cartésien partiel) + 1 pour id=2
    assert len(df_merged) == 3
    assert step.params["left_dupes"] == 1
    assert len(step.params["warnings"]) > 0


def test_merge_how_invalide(
    df_left: pd.DataFrame, df_right: pd.DataFrame
) -> None:
    """Test de fusion avec un type de jointure invalide."""
    with pytest.raises(InvalidMethodInputError, match="invalide"):
        merge_datasets(df_left, df_right, on="id", how="cross")


# ============================================================================
# Tests pour aggregate_by
# ============================================================================


def test_aggregate_by_cas_nominal() -> None:
    """Test d'agrégation simple."""
    df = pd.DataFrame(
        {
            "groupe": ["A", "A", "B", "B"],
            "score": [80, 90, 70, 75],
        }
    )
    df_agg, step = aggregate_by(df, "groupe", {"score": "mean"})
    assert len(df_agg) == 2
    assert step.operation == "aggregate"


def test_aggregate_by_colonne_inexistante() -> None:
    """Test d'agrégation avec une colonne inexistante."""
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(KeyError, match="n'existe pas"):
        aggregate_by(df, "a", {"b": "mean"})


# ============================================================================
# Tests pour pivot_longer
# ============================================================================


def test_pivot_longer_cas_nominal(df_wide: pd.DataFrame) -> None:
    """Test de pivot long."""
    df_long, step = pivot_longer(
        df_wide,
        id_vars=["id"],
        value_vars=["score_t1", "score_t2", "score_t3"],
    )
    assert len(df_long) == 9  # 3 ids × 3 temps
    assert "variable" in df_long.columns
    assert "value" in df_long.columns
    assert step.operation == "pivot_longer"


# ============================================================================
# Tests pour pivot_wider
# ============================================================================


def test_pivot_wider_cas_nominal(df_long: pd.DataFrame) -> None:
    """Test de pivot large."""
    df_wide, step = pivot_wider(
        df_long, index="id", columns="temps", values="score"
    )
    assert len(df_wide) == 3
    assert step.operation == "pivot_wider"


# ============================================================================
# Tests pour transpose_dataset
# ============================================================================


def test_transpose_cas_nominal() -> None:
    """Test de transposition."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df_t, step = transpose_dataset(df)
    assert len(df_t) == 2  # 2 colonnes → 2 lignes
    assert step.operation == "transpose"


# ============================================================================
# Tests pour drop_duplicates
# ============================================================================


def test_drop_duplicates_cas_nominal() -> None:
    """Test de dédoublonnage."""
    df = pd.DataFrame(
        {"id": [1, 1, 2, 2, 3], "val": ["a", "a", "b", "c", "d"]}
    )
    df_dedup, step = drop_duplicates(df, subset=["id", "val"])
    assert len(df_dedup) == 4
    assert step.params["n_removed"] == 1


def test_drop_duplicates_keep_invalide() -> None:
    """Test de dédoublonnage avec keep invalide."""
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(InvalidMethodInputError, match="invalide"):
        drop_duplicates(df, keep="middle")


# ============================================================================
# Tests pour sample_rows
# ============================================================================


def test_sample_rows_n() -> None:
    """Test d'échantillonnage par nombre."""
    df = pd.DataFrame({"a": range(100)})
    df_sample, step = sample_rows(df, n=10, random_state=42)
    assert len(df_sample) == 10
    assert step.params["random_state"] == 42


def test_sample_rows_frac() -> None:
    """Test d'échantillonnage par fraction."""
    df = pd.DataFrame({"a": range(100)})
    df_sample, _step = sample_rows(df, frac=0.2, random_state=42)
    assert len(df_sample) == 20


def test_sample_rows_reproductible() -> None:
    """Test que la même graine donne le même échantillon."""
    df = pd.DataFrame({"a": range(100)})
    df_s1, _ = sample_rows(df, n=10, random_state=42)
    df_s2, _ = sample_rows(df, n=10, random_state=42)
    pd.testing.assert_frame_equal(df_s1, df_s2)


def test_sample_rows_aucun_parametre() -> None:
    """Test d'échantillonnage sans n ni frac."""
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(InvalidMethodInputError, match="n.*frac"):
        sample_rows(df)


def test_sample_rows_deux_parametres() -> None:
    """Test d'échantillonnage avec n et frac simultanément."""
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(InvalidMethodInputError, match="pas les deux"):
        sample_rows(df, n=1, frac=0.5)


# ============================================================================
# Tests de non-mutation
# ============================================================================


def test_aucune_fonction_ne_mute_l_entree(
    df_left: pd.DataFrame, df_right: pd.DataFrame
) -> None:
    """Vérifie qu'aucune fonction ne modifie les DataFrames d'entrée."""
    df_l_orig = df_left.copy()
    df_r_orig = df_right.copy()

    merge_datasets(df_left, df_right, on="id")
    assert df_left.equals(df_l_orig)
    assert df_right.equals(df_r_orig)

    df = pd.DataFrame({"g": ["A", "B"], "v": [1, 2]})
    df_orig = df.copy()
    aggregate_by(df, "g", {"v": "mean"})
    assert df.equals(df_orig)

    drop_duplicates(df)
    assert df.equals(df_orig)

    sample_rows(df, n=1, random_state=0)
    assert df.equals(df_orig)
