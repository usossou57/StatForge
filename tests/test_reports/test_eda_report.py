"""Tests pour la fonction generate_eda_report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.models import DataDictionary, Variable
from app.reports.generator import generate_eda_report


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame de test pour le rapport EDA."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, n),
            "income": np.random.randn(n) * 10000 + 50000,
            "score": np.random.randn(n),
            "category": np.random.choice(["A", "B", "C"], n),
        }
    )


@pytest.fixture
def sample_data_dictionary() -> DataDictionary:
    """Dictionnaire de données de test."""
    return DataDictionary(
        dataset_name="test_dataset",
        n_rows=100,
        n_columns=4,
        variables={
            "age": Variable(name="age", dtype="numeric"),
            "income": Variable(name="income", dtype="numeric"),
            "score": Variable(name="score", dtype="numeric"),
            "category": Variable(name="category", dtype="categorical"),
        },
        source_path="test.csv",
        file_hash="abc123",
    )


def test_generate_eda_report_html(
    sample_df: pd.DataFrame, sample_data_dictionary: DataDictionary
) -> None:
    """Teste la génération d'un rapport EDA HTML."""
    report = generate_eda_report(
        sample_df, sample_data_dictionary, output_format="html"
    )
    assert isinstance(report, str)
    assert "<!DOCTYPE html>" in report
    assert "test_dataset" in report
    assert "Profil du dataset" in report
    assert "Valeurs manquantes" in report


def test_generate_eda_report_markdown(
    sample_df: pd.DataFrame, sample_data_dictionary: DataDictionary
) -> None:
    """Teste la génération d'un rapport EDA Markdown."""
    report = generate_eda_report(
        sample_df, sample_data_dictionary, output_format="markdown"
    )
    assert isinstance(report, str)
    assert "# Rapport EDA" in report
    assert "test_dataset" in report
    assert "## Profil du dataset" in report


def test_generate_eda_report_invalid_format(
    sample_df: pd.DataFrame, sample_data_dictionary: DataDictionary
) -> None:
    """Teste qu'un format invalide lève ValueError."""
    with pytest.raises(ValueError, match="non supporté"):
        generate_eda_report(sample_df, sample_data_dictionary, output_format="invalid")


def test_generate_eda_report_with_missing_values(
    sample_data_dictionary: DataDictionary,
) -> None:
    """Teste le rapport EDA avec des valeurs manquantes."""
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0, 5.0],
            "y": [10.0, np.nan, 30.0, 40.0, 50.0],
        }
    )
    report = generate_eda_report(df, sample_data_dictionary, output_format="html")
    assert "Valeurs manquantes" in report


def test_generate_eda_report_numeric_only(
    sample_data_dictionary: DataDictionary,
) -> None:
    """Teste le rapport EDA avec uniquement des variables numériques."""
    df = pd.DataFrame(
        {
            "a": np.random.randn(50),
            "b": np.random.randn(50),
            "c": np.random.randn(50),
        }
    )
    report = generate_eda_report(df, sample_data_dictionary, output_format="html")
    assert "Corrélations" in report
    assert "Tests de normalité" in report


def test_generate_eda_report_categorical_only(
    sample_data_dictionary: DataDictionary,
) -> None:
    """Teste le rapport EDA avec uniquement des variables catégorielles."""
    df = pd.DataFrame(
        {
            "cat1": np.random.choice(["A", "B", "C"], 50),
            "cat2": np.random.choice(["X", "Y"], 50),
        }
    )
    report = generate_eda_report(df, sample_data_dictionary, output_format="html")
    assert "Profil du dataset" in report
    # Pas de corrélations ni de tests de normalité pour des variables catégorielles
    assert "Corrélations" not in report or "Heatmap" not in report
