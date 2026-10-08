"""Tests pour le module app/statistics/descriptive.py."""

import numpy as np
import pandas as pd
import pytest

from app.results.models import AnalysisResult
from app.statistics.descriptive import (
    create_contingency_table,
    frequency_table,
    summarize_variables,
)


@pytest.fixture
def df_2x2() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sexe": ["H", "F", "H", "F", "H", "F"],
            "reussi": ["Oui", "Oui", "Non", "Oui", "Non", "Non"],
        }
    )


@pytest.fixture
def df_3x3() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "niveau": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "groupe": ["X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z"],
        }
    )


@pytest.fixture
def df_numerique_binaire() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "traitement": [0, 1, 0, 1, 0, 1],
            "effet": [0, 0, 1, 1, 0, 1],
        }
    )


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": [1.0, 2.0, 2.0, 4.0, np.nan, 6.0],
            "cat": ["A", "A", "B", "B", "B", "C"],
            "weights": [1.0, 2.0, 1.0, 1.0, 3.0, 2.0],
        }
    )


# ==============================================================================
# Tests pour create_contingency_table
# ==============================================================================
def test_cas_nominal_2x2(df_2x2: pd.DataFrame) -> None:
    result = create_contingency_table(df_2x2, "sexe", "reussi")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    assert result.loc["H", "Oui"] == 1
    assert result.values.sum() == 6


def test_cas_3x3(df_3x3: pd.DataFrame) -> None:
    result = create_contingency_table(df_3x3, "niveau", "groupe")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 3)
    assert result.loc["A", "X"] == 3


def test_variables_numeriques_binaires(df_numerique_binaire: pd.DataFrame) -> None:
    result = create_contingency_table(df_numerique_binaire, "traitement", "effet")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)


def test_colonne_manquante_row(df_2x2: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n'existe pas"):
        create_contingency_table(df_2x2, "inexistante", "reussi")


def test_colonne_manquante_col(df_2x2: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n'existe pas"):
        create_contingency_table(df_2x2, "sexe", "inexistante")


def test_pas_de_marges_par_defaut(df_2x2: pd.DataFrame) -> None:
    result = create_contingency_table(df_2x2, "sexe", "reussi")
    assert "All" not in result.index and "Total" not in result.index
    assert "All" not in result.columns and "Total" not in result.columns


def test_dtype_category_accepte() -> None:
    df = pd.DataFrame(
        {
            "couleur": pd.Categorical(["Rouge", "Bleu", "Rouge", "Vert"]),
            "forme": pd.Categorical(["Rond", "Carré", "Rond", "Carré"]),
        }
    )
    result = create_contingency_table(df, "couleur", "forme")
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 2)


# ==============================================================================
# Tests pour frequency_table
# ==============================================================================
def test_frequency_table_unweighted(sample_df: pd.DataFrame) -> None:
    result = frequency_table(sample_df, "cat")
    assert isinstance(result, AnalysisResult)
    freq_table = result.model_metrics["tables"]["frequency"]
    b_row = next(row for row in freq_table if row["value"] == "B")
    assert b_row["count"] == 3
    assert b_row["percentage"] == 50.0


def test_frequency_table_weighted(sample_df: pd.DataFrame) -> None:
    """Teste frequency_table avec pondération."""
    result = frequency_table(sample_df, "cat", weights="weights")
    assert isinstance(result, AnalysisResult)

    freq_table = result.model_metrics["tables"]["frequency"]
    # B a des poids aux indices 2, 3, 4 : 1.0 + 1.0 + 3.0 = 5.0. Poids total = 10.0
    b_row = next(row for row in freq_table if row["value"] == "B")
    assert b_row["count"] == 5.0
    assert b_row["percentage"] == 50.0


def test_frequency_table_colonne_manquante(sample_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n'existe pas"):
        frequency_table(sample_df, "inexistante")


def test_frequency_table_poids_manquants(sample_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n'existe pas"):
        frequency_table(sample_df, "cat", weights="inexistante")


# ==============================================================================
# Tests pour summarize_variables
# ==============================================================================
def test_summarize_variables_numeric(sample_df: pd.DataFrame) -> None:
    result = summarize_variables(sample_df, ["num"])
    assert isinstance(result, AnalysisResult)
    summary = result.model_metrics["tables"]["summary"][0]
    assert summary["variable"] == "num"
    assert summary["n"] == 5
    assert summary["n_missing"] == 1
    assert summary["missing_pct"] == pytest.approx(16.67, rel=1e-2)
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["median"] == pytest.approx(2.0)


def test_summarize_variables_categorical(sample_df: pd.DataFrame) -> None:
    result = summarize_variables(sample_df, ["cat"])
    assert isinstance(result, AnalysisResult)
    summary = result.model_metrics["tables"]["summary"][0]
    assert summary["variable"] == "cat"
    assert summary["n"] == 6
    assert summary["n_missing"] == 0
    assert summary["n_unique"] == 3
    assert summary["mode"] == "B"


def test_summarize_variables_default_columns(sample_df: pd.DataFrame) -> None:
    result = summarize_variables(sample_df)
    variables_analyzed = [
        row["variable"] for row in result.model_metrics["tables"]["summary"]
    ]
    assert "num" in variables_analyzed
    assert "weights" in variables_analyzed
    assert "cat" not in variables_analyzed
