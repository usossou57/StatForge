"""Tests pour le module de transformations de données."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data.transforms import (
    center,
    compute_variable,
    discretize,
    extract_date_parts,
    recode_values,
    standardize,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Retourne un DataFrame de test."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45],
            "score": [85.5, 92.0, 78.3, 88.7, 95.2],
            "groupe": ["A", "B", "A", "B", "A"],
            "date": pd.to_datetime(
                ["2024-01-15", "2024-02-20", "2024-03-25", "2024-04-30", "2024-05-05"]
            ),
        }
    )


# ============================================================================
# Tests pour recode_values
# ============================================================================


def test_recode_values_cas_nominal(sample_df: pd.DataFrame) -> None:
    """Test de recodage avec remplacement de la colonne."""
    df_original = sample_df.copy()
    mapping = {"A": "Groupe 1", "B": "Groupe 2"}
    df_new, step = recode_values(sample_df, "groupe", mapping)

    # Vérifier que l'entrée n'est pas modifiée
    assert sample_df.equals(df_original)
    # Vérifier le résultat
    assert df_new["groupe"].tolist() == [
        "Groupe 1",
        "Groupe 2",
        "Groupe 1",
        "Groupe 2",
        "Groupe 1",
    ]
    # Vérifier le TransformStep
    assert step.operation == "recode"
    assert step.params["mapping"] == mapping
    assert step.columns_in == ["groupe"]
    assert step.columns_out == ["groupe"]


def test_recode_values_nouvelle_colonne(sample_df: pd.DataFrame) -> None:
    """Test de recodage avec création d'une nouvelle colonne."""
    mapping = {"A": 1, "B": 2}
    df_new, step = recode_values(sample_df, "groupe", mapping, new_column="groupe_num")

    assert "groupe_num" in df_new.columns
    assert df_new["groupe_num"].tolist() == [1, 2, 1, 2, 1]
    assert "groupe" in df_new.columns  # Colonne originale préservée
    assert step.columns_out == ["groupe_num"]


def test_recode_values_colonne_inexistante(sample_df: pd.DataFrame) -> None:
    """Test de recodage avec une colonne inexistante."""
    with pytest.raises(KeyError, match="n'existe pas"):
        recode_values(sample_df, "inexistante", {"A": "B"})


# ============================================================================
# Tests pour compute_variable
# ============================================================================


def test_compute_variable_cas_nominal(sample_df: pd.DataFrame) -> None:
    """Test de calcul d'une nouvelle variable."""
    df_new, step = compute_variable(sample_df, "age_double", "age * 2")

    assert "age_double" in df_new.columns
    assert df_new["age_double"].tolist() == [50, 60, 70, 80, 90]
    assert step.operation == "compute"
    assert step.params["expression"] == "age * 2"


def test_compute_variable_expression_complexe(sample_df: pd.DataFrame) -> None:
    """Test de calcul avec une expression complexe."""
    df_new, _step = compute_variable(
        sample_df, "score_normalized", "(score - score.mean()) / score.std()"
    )

    assert "score_normalized" in df_new.columns
    # Vérifier que la moyenne est proche de 0 et l'écart-type proche de 1
    assert abs(df_new["score_normalized"].mean()) < 1e-10
    assert abs(df_new["score_normalized"].std() - 1.0) < 1e-10


def test_compute_variable_expression_invalide(sample_df: pd.DataFrame) -> None:
    """Test de calcul avec une expression non autorisée."""
    with pytest.raises(ValueError, match="non autorisé"):
        compute_variable(sample_df, "bad", "__import__('os').system('ls')")


def test_compute_variable_fonction_non_autorisee(sample_df: pd.DataFrame) -> None:
    """Test de calcul avec une fonction non autorisée."""
    with pytest.raises(ValueError, match="Fonction non autorisée"):
        compute_variable(sample_df, "bad", "exec('print(1)')")


# ============================================================================
# Tests pour standardize
# ============================================================================


def test_standardize_cas_nominal(sample_df: pd.DataFrame) -> None:
    """Test de standardisation."""
    df_new, step = standardize(sample_df, ["age", "score"])

    # Vérifier que la moyenne est proche de 0 et l'écart-type proche de 1
    assert abs(df_new["age"].mean()) < 1e-10
    assert abs(df_new["age"].std() - 1.0) < 1e-10
    assert abs(df_new["score"].mean()) < 1e-10
    assert abs(df_new["score"].std() - 1.0) < 1e-10
    assert step.operation == "standardize"
    assert step.columns_in == ["age", "score"]


def test_standardize_colonne_inexistante(sample_df: pd.DataFrame) -> None:
    """Test de standardisation avec une colonne inexistante."""
    with pytest.raises(KeyError, match="n'existe pas"):
        standardize(sample_df, ["age", "inexistante"])


# ============================================================================
# Tests pour center
# ============================================================================


def test_center_cas_nominal(sample_df: pd.DataFrame) -> None:
    """Test de centrage."""
    df_new, step = center(sample_df, ["age", "score"])

    # Vérifier que la moyenne est proche de 0
    assert abs(df_new["age"].mean()) < 1e-10
    assert abs(df_new["score"].mean()) < 1e-10
    assert step.operation == "center"


def test_center_colonne_inexistante(sample_df: pd.DataFrame) -> None:
    """Test de centrage avec une colonne inexistante."""
    with pytest.raises(KeyError, match="n'existe pas"):
        center(sample_df, ["inexistante"])


# ============================================================================
# Tests pour discretize
# ============================================================================


def test_discretize_width_cas_nominal(sample_df: pd.DataFrame) -> None:
    """Test de discrétisation par largeur égale."""
    df_new, step = discretize(sample_df, "age", bins=3, strategy="width")

    assert df_new["age"].dtype.name == "category"
    assert len(df_new["age"].cat.categories) == 3
    assert step.operation == "discretize"
    assert step.params["strategy"] == "width"


def test_discretize_quantile(sample_df: pd.DataFrame) -> None:
    """Test de discrétisation par quantiles."""
    df_new, _step = discretize(sample_df, "score", bins=2, strategy="quantile")

    assert df_new["score"].dtype.name == "category"
    assert len(df_new["score"].cat.categories) == 2


def test_discretize_bornes_explicites(sample_df: pd.DataFrame) -> None:
    """Test de discrétisation avec bornes explicites."""
    df_new, _step = discretize(sample_df, "age", bins=[20, 30, 40, 50])

    assert df_new["age"].dtype.name == "category"
    assert len(df_new["age"].cat.categories) == 3


def test_discretize_labels_personnalises(sample_df: pd.DataFrame) -> None:
    """Test de discrétisation avec labels personnalisés."""
    df_new, _step = discretize(
        sample_df, "age", bins=3, labels=["Jeune", "Adulte", "Senior"], strategy="width"
    )

    assert set(df_new["age"].cat.categories) == {"Jeune", "Adulte", "Senior"}


def test_discretize_strategie_invalide(sample_df: pd.DataFrame) -> None:
    """Test de discrétisation avec une stratégie invalide."""
    with pytest.raises(ValueError, match="Stratégie non supportée"):
        discretize(sample_df, "age", bins=3, strategy="invalid")


# ============================================================================
# Tests pour extract_date_parts
# ============================================================================


def test_extract_date_parts_cas_nominal(sample_df: pd.DataFrame) -> None:
    """Test d'extraction de parties de date."""
    df_new, step = extract_date_parts(sample_df, "date", ["year", "month", "day"])

    assert "date_year" in df_new.columns
    assert "date_month" in df_new.columns
    assert "date_day" in df_new.columns
    assert df_new["date_year"].tolist() == [2024, 2024, 2024, 2024, 2024]
    assert df_new["date_month"].tolist() == [1, 2, 3, 4, 5]
    assert df_new["date_day"].tolist() == [15, 20, 25, 30, 5]
    assert step.operation == "extract_date_parts"
    assert step.columns_out == ["date_year", "date_month", "date_day"]


def test_extract_date_parts_partie_invalide(sample_df: pd.DataFrame) -> None:
    """Test d'extraction avec une partie non supportée."""
    with pytest.raises(ValueError, match="Partie non supportée"):
        extract_date_parts(sample_df, "date", ["invalid"])


def test_extract_date_parts_colonne_inexistante(sample_df: pd.DataFrame) -> None:
    """Test d'extraction avec une colonne inexistante."""
    with pytest.raises(KeyError, match="n'existe pas"):
        extract_date_parts(sample_df, "inexistante", ["year"])


# ============================================================================
# Tests de non-mutation
# ============================================================================


def test_aucune_fonction_ne_mute_l_entree(sample_df: pd.DataFrame) -> None:
    """Vérifie qu'aucune fonction ne modifie le DataFrame d'entrée."""
    df_original = sample_df.copy()

    # Tester chaque fonction
    recode_values(sample_df, "groupe", {"A": "X"})
    assert sample_df.equals(df_original)

    compute_variable(sample_df, "new", "age * 2")
    assert sample_df.equals(df_original)

    standardize(sample_df, ["age"])
    assert sample_df.equals(df_original)

    center(sample_df, ["age"])
    assert sample_df.equals(df_original)

    discretize(sample_df, "age", bins=3)
    assert sample_df.equals(df_original)

    extract_date_parts(sample_df, "date", ["year"])
    assert sample_df.equals(df_original)


# ============================================================================
# Tests de rejeu via TransformLog
# ============================================================================


def test_transformations_rejouables_via_log(sample_df: pd.DataFrame) -> None:
    """Vérifie que les TransformStep générés permettent de rejouer les transformations."""
    from app.data.transform_log import TransformLog

    log = TransformLog()

    # Appliquer une séquence de transformations
    df1, step1 = recode_values(sample_df, "groupe", {"A": 1, "B": 2})
    log.add(step1)

    df2, step2 = center(df1, ["age"])
    log.add(step2)

    _df3, step3 = standardize(df2, ["score"])
    log.add(step3)

    # Générer le script
    script = log.to_python_script()

    # Vérifier que le script contient les opérations
    assert "map({1: 1, 2: 2})" in script or "map({'A': 1, 'B': 2})" in script
    assert "age" in script
    assert "score" in script
