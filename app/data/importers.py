"""Module d'import de fichiers de données.

Ce module fournit des fonctions pour importer des fichiers de données
de différents formats (CSV, Excel, SPSS) et les convertir en DataFrames pandas.
"""

from pathlib import Path

import pandas as pd
import pyreadstat


def import_csv(path: str) -> pd.DataFrame:
    """Importe un fichier CSV et retourne un DataFrame pandas.

    Args:
        path: Chemin vers le fichier CSV à importer.

    Returns:
        DataFrame pandas contenant les données du fichier CSV.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le fichier ne peut pas être lu ou n'est pas un CSV valide.

    Example:
        >>> df = import_csv("/chemin/vers/donnees.csv")
        >>> print(df.shape)
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Le fichier n'existe pas : {path}")

    try:
        return pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Impossible de lire le fichier CSV : {path}. Erreur : {e}") from e


def import_xlsx(path: str) -> pd.DataFrame:
    """Importe un fichier Excel (.xlsx) et retourne un DataFrame pandas.

    Args:
        path: Chemin vers le fichier Excel (.xlsx) à importer.

    Returns:
        DataFrame pandas contenant les données du fichier Excel.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le fichier ne peut pas être lu ou n'est pas un Excel valide.

    Example:
        >>> df = import_xlsx("/chemin/vers/donnees.xlsx")
        >>> print(df.shape)
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Le fichier n'existe pas : {path}")

    try:
        return pd.read_excel(path)
    except Exception as e:
        raise ValueError(
            f"Impossible de lire le fichier Excel : {path}. Erreur : {e}"
        ) from e


def import_sav(path: str) -> tuple[pd.DataFrame, dict]:
    """Importe un fichier SPSS (.sav) et retourne (df, labels).

    Args:
        path: Chemin vers le fichier .sav.

    Returns:
        Un tuple contenant :
        - df: Le DataFrame pandas avec les données.
        - labels: Un dictionnaire des labels de valeurs extraits via pyreadstat
          (meta.variable_value_labels).

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le fichier ne peut pas être lu ou n'est pas un fichier SPSS valide.

    Example:
        >>> df, labels = import_sav("/chemin/vers/donnees.sav")
        >>> print(df.shape)
        >>> print(labels)
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Le fichier n'existe pas : {path}")

    try:
        df, meta = pyreadstat.read_sav(path)
        labels = meta.variable_value_labels
        return df, labels
    except Exception as e:
        raise ValueError(
            f"Impossible de lire le fichier SPSS : {path}. Erreur : {e}"
        ) from e

