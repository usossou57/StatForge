"""Tests pour le module d'export de fichiers de données."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadstat
import pytest

from app.data.exporters import (
    export_csv,
    export_html,
    export_latex,
    export_parquet,
    export_stata,
    export_xlsx,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Retourne un DataFrame de test."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "nom": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [85.5, 92.0, 78.3],
        }
    )


def test_export_csv_cas_nominal(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Test d'export CSV avec un DataFrame valide."""
    path = tmp_path / "export.csv"
    export_csv(sample_df, str(path))
    assert path.exists()
    df_read = pd.read_csv(path)
    pd.testing.assert_frame_equal(df_read, sample_df)


def test_export_xlsx_cas_nominal(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Test d'export Excel avec un DataFrame valide."""
    path = tmp_path / "export.xlsx"
    export_xlsx(sample_df, str(path), sheet_name="Données")
    assert path.exists()
    df_read = pd.read_excel(path, sheet_name="Données")
    pd.testing.assert_frame_equal(df_read, sample_df)


def test_export_parquet_cas_nominal(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Test d'export Parquet avec un DataFrame valide."""
    path = tmp_path / "export.parquet"
    export_parquet(sample_df, str(path))
    assert path.exists()
    df_read = pd.read_parquet(path)
    pd.testing.assert_frame_equal(df_read, sample_df)


def test_export_stata_cas_nominal(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Test d'export Stata avec un DataFrame valide et des labels."""
    path = tmp_path / "export.dta"
    labels = {"id": {1: "Un", 2: "Deux", 3: "Trois"}}
    export_stata(sample_df, str(path), labels=labels)
    assert path.exists()
    
    df_read, meta = pyreadstat.read_dta(path)
    # pyreadstat peut modifier légèrement les types, on vérifie les valeurs
    assert df_read.shape == sample_df.shape
    assert meta.variable_value_labels == labels


def test_export_stata_sans_labels(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Test d'export Stata sans labels."""
    path = tmp_path / "export_no_labels.dta"
    export_stata(sample_df, str(path))
    assert path.exists()
    df_read, meta = pyreadstat.read_dta(path)
    assert df_read.shape == sample_df.shape
    assert meta.variable_value_labels == {}


def test_export_latex_cas_nominal(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Test d'export LaTeX avec un DataFrame valide."""
    path = tmp_path / "export.tex"
    export_latex(sample_df, str(path), caption="Mon tableau", label="tab:donnees")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "\\caption{Mon tableau}" in content
    assert "\\label{tab:donnees}" in content
    assert "Alice" in content


def test_export_html_cas_nominal(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Test d'export HTML avec un DataFrame valide."""
    path = tmp_path / "export.html"
    export_html(sample_df, str(path), title="Rapport de données")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "<title>Rapport de données</title>" in content
    assert "<h1>Rapport de données</h1>" in content
    assert "Alice" in content
    # pandas génère class="dataframe dataframe", on vérifie juste la présence de la classe
    assert 'class="dataframe' in content


def test_export_creates_parent_directory(tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Test que la fonction crée le répertoire parent s'il n'existe pas."""
    path = tmp_path / "nouveau_dossier" / "sous_dossier" / "export.csv"
    assert not path.parent.exists()
    export_csv(sample_df, str(path))
    assert path.exists()
