"""Tests pour le module d'import de fichiers de données.

Ce module teste les fonctions d'import pour les formats CSV, Excel (XLSX),
SPSS (SAV), Stata (DTA), SAS (sas7bdat), JSON, Parquet, XML et SQL.
Inclut les cas nominaux et les cas d'erreur.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.data.importers import (
    import_csv,
    import_json,
    import_parquet,
    import_sas,
    import_sav,
    import_sql,
    import_stata,
    import_xlsx,
    import_xml,
)
from app.errors import InvalidMethodInputError

# Chemin vers le répertoire des fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def sample_csv_path() -> str:
    """Retourne le chemin vers le fichier CSV de test."""
    return str(FIXTURES_DIR / "sample.csv")


@pytest.fixture
def sample_xlsx_path(tmp_path: Path) -> str:
    """Crée un fichier XLSX de test et retourne son chemin."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "nom": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [85.5, 92.0, 78.3],
        }
    )
    xlsx_path = tmp_path / "sample.xlsx"
    df.to_excel(xlsx_path, index=False)
    return str(xlsx_path)


@pytest.fixture
def sample_json_path(tmp_path: Path) -> str:
    """Crée un fichier JSON de test et retourne son chemin."""
    data = [
        {"id": 1, "nom": "Alice", "age": 25},
        {"id": 2, "nom": "Bob", "age": 30},
        {"id": 3, "nom": "Charlie", "age": 35},
    ]
    json_path = tmp_path / "sample.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return str(json_path)


@pytest.fixture
def sample_jsonl_path(tmp_path: Path) -> str:
    """Crée un fichier JSON Lines de test et retourne son chemin."""
    data = [
        {"id": 1, "nom": "Alice", "age": 25},
        {"id": 2, "nom": "Bob", "age": 30},
    ]
    jsonl_path = tmp_path / "sample.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in data)
    return str(jsonl_path)


@pytest.fixture
def sample_parquet_path(tmp_path: Path) -> str:
    """Crée un fichier Parquet de test et retourne son chemin."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "nom": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
        }
    )
    parquet_path = tmp_path / "sample.parquet"
    df.to_parquet(parquet_path)
    return str(parquet_path)


@pytest.fixture
def sample_xml_path(tmp_path: Path) -> str:
    """Crée un fichier XML de test et retourne son chemin."""
    xml_content = """<?xml version='1.0' encoding='utf-8'?>
<data>
  <row>
    <id>1</id>
    <nom>Alice</nom>
    <age>25</age>
  </row>
  <row>
    <id>2</id>
    <nom>Bob</nom>
    <age>30</age>
  </row>
</data>"""
    xml_path = tmp_path / "sample.xml"
    xml_path.write_text(xml_content, encoding="utf-8")
    return str(xml_path)


@pytest.fixture
def sample_sqlite_path(tmp_path: Path) -> str:
    """Crée une base SQLite de test et retourne son URL de connexion."""
    db_path = tmp_path / "test.db"
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "nom": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
        }
    )
    # Import paresseux de sqlalchemy
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql("users", engine, index=False)
    return f"sqlite:///{db_path}"


# ============================================================================
# Tests pour import_csv
# ============================================================================


def test_import_csv_cas_nominal(sample_csv_path: str) -> None:
    """Test d'import CSV avec un fichier valide."""
    df = import_csv(sample_csv_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (5, 4)
    assert list(df.columns) == ["id", "nom", "age", "score"]
    assert df.iloc[0]["nom"] == "Alice"
    assert df.iloc[0]["age"] == 25
    assert df.iloc[4]["score"] == 95.2


def test_import_csv_fichier_inexistant() -> None:
    """Test d'import CSV avec un fichier inexistant."""
    with pytest.raises(FileNotFoundError) as exc_info:
        import_csv("/chemin/inexistant/fichier.csv")
    assert "n'existe pas" in str(exc_info.value)


def test_import_csv_format_invalide(tmp_path: Path) -> None:
    """Test d'import CSV avec un fichier au format invalide (vide)."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="Impossible de lire le fichier CSV"):
        import_csv(str(empty_file))


# ============================================================================
# Tests pour import_xlsx
# ============================================================================


def test_import_xlsx_cas_nominal(sample_xlsx_path: str) -> None:
    """Test d'import Excel avec un fichier valide."""
    df = import_xlsx(sample_xlsx_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 4)
    assert list(df.columns) == ["id", "nom", "age", "score"]
    assert df.iloc[0]["nom"] == "Alice"
    assert df.iloc[1]["age"] == 30


def test_import_xlsx_fichier_inexistant() -> None:
    """Test d'import Excel avec un fichier inexistant."""
    with pytest.raises(FileNotFoundError) as exc_info:
        import_xlsx("/chemin/inexistant/fichier.xlsx")
    assert "n'existe pas" in str(exc_info.value)


# ============================================================================
# Tests pour import_sav
# ============================================================================


def test_import_sav_cas_nominal(tmp_path: Path) -> None:
    """Test d'import SPSS avec un fichier valide (mocké)."""
    mock_df = pd.DataFrame(
        {
            "sexe": [1, 2, 1, 2, 1],
            "age": [25, 30, 35, 28, 32],
            "score": [85.5, 92.0, 78.3, 88.7, 95.2],
        }
    )
    mock_meta = MagicMock()
    mock_meta.variable_value_labels = {
        "sexe": {1: "Homme", 2: "Femme"},
    }

    dummy_sav = tmp_path / "donnees.sav"
    dummy_sav.touch()

    with patch("app.data.importers.pyreadstat.read_sav") as mock_read_sav:
        mock_read_sav.return_value = (mock_df, mock_meta)
        df, labels = import_sav(str(dummy_sav))
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (5, 3)
        assert list(df.columns) == ["sexe", "age", "score"]
        assert isinstance(labels, dict)
        assert "sexe" in labels
        assert labels["sexe"] == {1: "Homme", 2: "Femme"}
        mock_read_sav.assert_called_once_with(str(dummy_sav))


def test_import_sav_fichier_inexistant() -> None:
    """Test d'import SPSS avec un fichier inexistant."""
    with pytest.raises(FileNotFoundError) as exc_info:
        import_sav("/chemin/inexistant/fichier.sav")
    assert "n'existe pas" in str(exc_info.value)


def test_import_sav_labels_extraits(tmp_path: Path) -> None:
    """Test que les labels de valeurs sont correctement extraits du fichier SPSS."""
    mock_df = pd.DataFrame(
        {
            "sexe": [1, 2, 1, 1, 2],
            "niveau_etude": [1, 2, 3, 1, 2],
            "satisfaction": [1, 2, 3, 4, 5],
        }
    )
    mock_meta = MagicMock()
    mock_meta.variable_value_labels = {
        "sexe": {1: "Homme", 2: "Femme"},
        "niveau_etude": {
            1: "Primaire",
            2: "Secondaire",
            3: "Supérieur",
        },
        "satisfaction": {
            1: "Très insatisfait",
            2: "Insatisfait",
            3: "Neutre",
            4: "Satisfait",
            5: "Très satisfait",
        },
    }

    dummy_sav = tmp_path / "enquete.sav"
    dummy_sav.touch()

    with patch("app.data.importers.pyreadstat.read_sav") as mock_read_sav:
        mock_read_sav.return_value = (mock_df, mock_meta)
        _df, labels = import_sav(str(dummy_sav))
        assert len(labels) == 3
        assert labels["sexe"] == {1: "Homme", 2: "Femme"}
        assert labels["niveau_etude"][2] == "Secondaire"
        assert labels["satisfaction"][5] == "Très satisfait"


def test_import_sav_erreur_lecture(tmp_path: Path) -> None:
    """Test d'import SPSS avec un fichier corrompu ou invalide."""
    invalid_file = tmp_path / "invalid.sav"
    invalid_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")

    with patch("app.data.importers.pyreadstat.read_sav") as mock_read_sav:
        mock_read_sav.side_effect = Exception("Fichier SPSS corrompu")
        with pytest.raises(ValueError) as exc_info:
            import_sav(str(invalid_file))
        assert "Impossible de lire le fichier SPSS" in str(exc_info.value)


# ============================================================================
# Tests pour import_json
# ============================================================================


def test_import_json_cas_nominal(sample_json_path: str) -> None:
    """Test d'import JSON avec un fichier valide (JSON array)."""
    df = import_json(sample_json_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert list(df.columns) == ["id", "nom", "age"]
    assert df.iloc[0]["nom"] == "Alice"
    assert df.iloc[2]["age"] == 35


def test_import_json_jsonl_cas_nominal(sample_jsonl_path: str) -> None:
    """Test d'import JSON Lines avec un fichier valide."""
    df = import_json(sample_jsonl_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["id", "nom", "age"]
    assert df.iloc[0]["nom"] == "Alice"


def test_import_json_fichier_inexistant() -> None:
    """Test d'import JSON avec un fichier inexistant."""
    with pytest.raises(FileNotFoundError) as exc_info:
        import_json("/chemin/inexistant/fichier.json")
    assert "n'existe pas" in str(exc_info.value)


def test_import_json_format_invalide(tmp_path: Path) -> None:
    """Test d'import JSON avec un fichier au format invalide."""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("ceci n'est pas du json")
    with pytest.raises(ValueError, match="Impossible de lire le fichier JSON"):
        import_json(str(invalid_file))


# ============================================================================
# Tests pour import_parquet
# ============================================================================


def test_import_parquet_cas_nominal(sample_parquet_path: str) -> None:
    """Test d'import Parquet avec un fichier valide."""
    df = import_parquet(sample_parquet_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert list(df.columns) == ["id", "nom", "age"]
    assert df.iloc[0]["nom"] == "Alice"
    assert df.iloc[1]["age"] == 30


def test_import_parquet_fichier_inexistant() -> None:
    """Test d'import Parquet avec un fichier inexistant."""
    with pytest.raises(FileNotFoundError) as exc_info:
        import_parquet("/chemin/inexistant/fichier.parquet")
    assert "n'existe pas" in str(exc_info.value)


# ============================================================================
# Tests pour import_stata
# ============================================================================


def test_import_stata_cas_nominal(tmp_path: Path) -> None:
    """Test d'import Stata avec un fichier valide (mocké)."""
    mock_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "score": [85.5, 92.0, 78.3],
        }
    )
    mock_meta = MagicMock()
    mock_meta.variable_value_labels = {}

    dummy_dta = tmp_path / "donnees.dta"
    dummy_dta.touch()

    with patch("app.data.importers.pyreadstat.read_dta") as mock_read_dta:
        mock_read_dta.return_value = (mock_df, mock_meta)
        df, labels = import_stata(str(dummy_dta))
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert list(df.columns) == ["id", "score"]
        assert isinstance(labels, dict)
        mock_read_dta.assert_called_once_with(str(dummy_dta))


def test_import_stata_fichier_inexistant() -> None:
    """Test d'import Stata avec un fichier inexistant."""
    with pytest.raises(FileNotFoundError) as exc_info:
        import_stata("/chemin/inexistant/fichier.dta")
    assert "n'existe pas" in str(exc_info.value)


# ============================================================================
# Tests pour import_sas
# ============================================================================


def test_import_sas_cas_nominal(tmp_path: Path) -> None:
    """Test d'import SAS avec un fichier valide (mocké)."""
    mock_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10.5, 20.3, 30.1],
        }
    )
    mock_meta = MagicMock()
    mock_meta.variable_value_labels = {}

    dummy_sas = tmp_path / "donnees.sas7bdat"
    dummy_sas.touch()

    with patch("app.data.importers.pyreadstat.read_sas7bdat") as mock_read_sas:
        mock_read_sas.return_value = (mock_df, mock_meta)
        df, labels = import_sas(str(dummy_sas))
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert list(df.columns) == ["id", "value"]
        assert isinstance(labels, dict)
        mock_read_sas.assert_called_once_with(str(dummy_sas))


def test_import_sas_fichier_inexistant() -> None:
    """Test d'import SAS avec un fichier inexistant."""
    with pytest.raises(FileNotFoundError) as exc_info:
        import_sas("/chemin/inexistant/fichier.sas7bdat")
    assert "n'existe pas" in str(exc_info.value)


# ============================================================================
# Tests pour import_xml
# ============================================================================


def test_import_xml_cas_nominal(sample_xml_path: str) -> None:
    """Test d'import XML avec un fichier valide."""
    df = import_xml(sample_xml_path, xpath=".//row")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["id", "nom", "age"]
    assert df.iloc[0]["nom"] == "Alice"
    assert df.iloc[1]["age"] == 30


def test_import_xml_fichier_inexistant() -> None:
    """Test d'import XML avec un fichier inexistant."""
    with pytest.raises(FileNotFoundError) as exc_info:
        import_xml("/chemin/inexistant/fichier.xml")
    assert "n'existe pas" in str(exc_info.value)


def test_import_xml_xpath_invalide(tmp_path: Path) -> None:
    """Test d'import XML avec un XPath invalide."""
    xml_file = tmp_path / "test.xml"
    xml_file.write_text("<data><row><id>1</id></row></data>")
    with pytest.raises(ValueError, match="Impossible de lire le fichier XML"):
        import_xml(str(xml_file), xpath="//invalid/path/that/does/not/exist")


# ============================================================================
# Tests pour import_sql
# ============================================================================


def test_import_sql_cas_nominal(sample_sqlite_path: str) -> None:
    """Test d'import SQL avec une requête SELECT valide."""
    df = import_sql(sample_sqlite_path, "SELECT * FROM users")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert list(df.columns) == ["id", "nom", "age"]
    assert df.iloc[0]["nom"] == "Alice"
    assert df.iloc[2]["age"] == 35


def test_import_sql_avec_where(sample_sqlite_path: str) -> None:
    """Test d'import SQL avec une requête SELECT avec WHERE."""
    df = import_sql(sample_sqlite_path, "SELECT * FROM users WHERE age > 25")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert all(df["age"] > 25)


def test_import_sql_requete_non_select() -> None:
    """Test qu'une requête non-SELECT lève InvalidMethodInputError."""
    with pytest.raises(InvalidMethodInputError) as exc_info:
        import_sql("sqlite:///test.db", "INSERT INTO users VALUES (4, 'Dave', 40)")
    assert "SELECT" in str(exc_info.value)
    assert "autorisées" in str(exc_info.value)


def test_import_sql_requete_dangereuse() -> None:
    """Test qu'une requête avec des mots-clés dangereux est rejetée."""
    dangerous_queries = [
        "SELECT * FROM users; DROP TABLE users",
        "SELECT * FROM users WHERE 1=1; DELETE FROM users",
        "SELECT * FROM users; UPDATE users SET age=99",
    ]
    for query in dangerous_queries:
        with pytest.raises(InvalidMethodInputError):
            import_sql("sqlite:///test.db", query)


def test_import_sql_connexion_invalide() -> None:
    """Test d'import SQL avec une URL de connexion invalide."""
    with pytest.raises(ValueError, match="Impossible d'exécuter la requête SQL"):
        import_sql("sqlite:///chemin/inexistant/base.db", "SELECT 1")


def test_import_sql_requete_invalide(sample_sqlite_path: str) -> None:
    """Test d'import SQL avec une requête SQL invalide."""
    with pytest.raises(ValueError, match="Impossible d'exécuter la requête SQL"):
        import_sql(sample_sqlite_path, "SELECT * FROM table_inexistante")