"""Tests pour le module d'import de fichiers de données.

Ce module teste les fonctions d'import pour les formats CSV, Excel (XLSX)
et SPSS (SAV), incluant les cas nominaux et les cas d'erreur.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.data.importers import import_csv, import_sav, import_xlsx

# Chemin vers le répertoire des fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def sample_csv_path() -> str:
    """Retourne le chemin vers le fichier CSV de test."""
    return str(FIXTURES_DIR / "sample.csv")


@pytest.fixture
def sample_xlsx_path(tmp_path: Path) -> str:
    """Crée un fichier XLSX de test et retourne son chemin.

    Le fichier est créé dynamiquement avec pandas pour garantir
    sa validité et sa portabilité.
    """
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


# === Tests pour import_csv ===


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

    # Un fichier vide doit lever une ValueError (format invalide)
    with pytest.raises(ValueError, match="Impossible de lire le fichier CSV"):
        import_csv(str(empty_file))


# === Tests pour import_xlsx ===


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


# === Tests pour import_sav ===


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

    # Créer un fichier fictif pour que file_path.exists() soit True
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
    # CORRECTION : Toutes les listes doivent avoir la même longueur (ici 5)
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

    # Créer un fichier fictif pour que file_path.exists() soit True
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
