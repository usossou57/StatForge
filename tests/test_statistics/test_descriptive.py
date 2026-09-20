"""Tests pour le module app/statistics/descriptive.py.

Ces tests vérifient le comportement de la fonction create_contingency_table
dans différents cas : nominal, multi-modalités, et cas limites (erreurs).
"""

import pandas as pd
import pytest

from app.statistics.descriptive import create_contingency_table


@pytest.fixture
def df_2x2() -> pd.DataFrame:
    """DataFrame simple avec deux variables binaires."""
    return pd.DataFrame(
        {
            "sexe": ["H", "F", "H", "F", "H", "F"],
            "reussi": ["Oui", "Oui", "Non", "Oui", "Non", "Non"],
        }
    )


@pytest.fixture
def df_3x3() -> pd.DataFrame:
    """DataFrame avec deux variables à 3 modalités."""
    return pd.DataFrame(
        {
            "niveau": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "groupe": ["X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z"],
        }
    )


@pytest.fixture
def df_numerique_binaire() -> pd.DataFrame:
    """DataFrame avec variables numériques encodées (0/1)."""
    return pd.DataFrame(
        {
            "traitement": [0, 1, 0, 1, 0, 1],
            "effet": [0, 0, 1, 1, 0, 1],
        }
    )


@pytest.fixture
def df_continu() -> pd.DataFrame:
    """DataFrame avec une variable continue (non catégorielle)."""
    return pd.DataFrame(
        {
            "age": [25.5, 30.1, 45.7, 22.3, 38.9, 51.2, 29.4, 33.6, 42.1, 27.8, 55.0],
            "groupe": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A"],
        }
    )


def test_cas_nominal_2x2(df_2x2: pd.DataFrame) -> None:
    """Teste le calcul d'un tableau de contingence 2x2."""
    result = create_contingency_table(df_2x2, "sexe", "reussi")

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    assert list(result.index) == ["F", "H"]
    assert list(result.columns) == ["Non", "Oui"]
    # Vérification des effectifs
    assert result.loc["H", "Oui"] == 1
    assert result.loc["H", "Non"] == 2
    assert result.loc["F", "Oui"] == 2
    assert result.loc["F", "Non"] == 1
    # Somme totale = nombre d'observations
    assert result.values.sum() == 6


def test_cas_3x3(df_3x3: pd.DataFrame) -> None:
    """Teste le calcul d'un tableau de contingence 3x3."""
    result = create_contingency_table(df_3x3, "niveau", "groupe")

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)
    assert list(result.index) == ["A", "B", "C"]
    assert list(result.columns) == ["X", "Y", "Z"]
    
    # Le fixture contient 3 occurrences de chaque paire (A,X), (B,Y), (C,Z)
    assert result.loc["A", "X"] == 3
    assert result.loc["B", "Y"] == 3
    assert result.loc["C", "Z"] == 3


def test_variables_numeriques_binaires(df_numerique_binaire: pd.DataFrame) -> None:
    """Teste que les variables numériques avec peu de modalités sont acceptées."""
    result = create_contingency_table(
        df_numerique_binaire, "traitement", "effet"
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    assert result.values.sum() == 6


def test_colonne_manquante_row(df_2x2: pd.DataFrame) -> None:
    """Teste que ValueError est levée si la colonne row n'existe pas."""
    with pytest.raises(ValueError, match="n'existe pas"):
        create_contingency_table(df_2x2, "inexistante", "reussi")


def test_colonne_manquante_col(df_2x2: pd.DataFrame) -> None:
    """Teste que ValueError est levée si la colonne col n'existe pas."""
    with pytest.raises(ValueError, match="n'existe pas"):
        create_contingency_table(df_2x2, "sexe", "inexistante")


def test_variable_non_categorielle_row(df_continu: pd.DataFrame) -> None:
    """Teste que ValueError est levée si row est une variable continue."""
    with pytest.raises(ValueError, match="n'est pas catégorielle"):
        create_contingency_table(df_continu, "age", "groupe")


def test_variable_non_categorielle_col(df_continu: pd.DataFrame) -> None:
    """Teste que ValueError est levée si col est une variable continue."""
    with pytest.raises(ValueError, match="n'est pas catégorielle"):
        create_contingency_table(df_continu, "groupe", "age")


def test_pas_de_marges_par_defaut(df_2x2: pd.DataFrame) -> None:
    """Vérifie que le tableau retourné ne contient pas de marges (All)."""
    result = create_contingency_table(df_2x2, "sexe", "reussi")

    assert "All" not in result.index
    assert "All" not in result.columns


def test_dtype_category_accepte() -> None:
    """Teste que les colonnes de type category sont acceptées."""
    df = pd.DataFrame(
        {
            "couleur": pd.Categorical(["Rouge", "Bleu", "Rouge", "Vert"]),
            "forme": pd.Categorical(["Rond", "Carré", "Rond", "Carré"]),
        }
    )
    result = create_contingency_table(df, "couleur", "forme")

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 3  # 3 couleurs
    assert result.shape[1] == 2  # 2 formes

