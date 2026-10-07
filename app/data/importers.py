"""Module d'import de fichiers de données.

Ce module fournit des fonctions pour importer des fichiers de données
de différents formats (CSV, Excel, SPSS, SAS, Stata, JSON, Parquet, XML, SQL)
et les convertir en DataFrames pandas.

V2 : ajout des formats JSON, Parquet, Stata, SAS, XML et SQL.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadstat

from app.errors import InvalidMethodInputError


def import_csv(path: str) -> pd.DataFrame:
    """Importe un fichier CSV et retourne un DataFrame pandas.

    Args:
        path: Chemin vers le fichier CSV à importer.

    Returns:
        DataFrame pandas contenant les données du fichier CSV.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le fichier ne peut pas être lu ou n'est pas un CSV valide.
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


def import_json(path: str) -> pd.DataFrame:
    """Importe un fichier JSON et retourne un DataFrame pandas.

    Supporte deux formats :
    - JSON array : [{"col1": val1, "col2": val2}, ...]
    - JSON Lines : une ligne JSON par enregistrement

    Args:
        path: Chemin vers le fichier JSON à importer.

    Returns:
        DataFrame pandas contenant les données du fichier JSON.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le fichier ne peut pas être lu ou n'est pas un JSON valide.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Le fichier n'existe pas : {path}")
    try:
        # Tente d'abord un JSON array classique
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Détection du format : JSON Lines si plusieurs lignes commençant par {
        if content.startswith("{") and "\n" in content:
            # Format JSON Lines
            return pd.read_json(file_path, lines=True, encoding="utf-8")
        else:
            # Format JSON array ou objet
            return pd.read_json(file_path, encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Impossible de lire le fichier JSON : {path}. Erreur : {e}") from e


def import_parquet(path: str) -> pd.DataFrame:
    """Importe un fichier Parquet et retourne un DataFrame pandas.

    Nécessite la dépendance pyarrow (installée par défaut dans la Phase 15).

    Args:
        path: Chemin vers le fichier Parquet à importer.

    Returns:
        DataFrame pandas contenant les données du fichier Parquet.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le fichier ne peut pas être lu ou n'est pas un Parquet valide.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Le fichier n'existe pas : {path}")
    try:
        return pd.read_parquet(path)
    except Exception as e:
        raise ValueError(
            f"Impossible de lire le fichier Parquet : {path}. Erreur : {e}"
        ) from e


def import_stata(path: str) -> tuple[pd.DataFrame, dict]:
    """Importe un fichier Stata (.dta) et retourne (df, labels).

    Args:
        path: Chemin vers le fichier .dta.

    Returns:
        Un tuple contenant :
        - df: Le DataFrame pandas avec les données.
        - labels: Un dictionnaire des labels de valeurs extraits via pyreadstat
          (meta.variable_value_labels).

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le fichier ne peut pas être lu ou n'est pas un fichier Stata valide.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Le fichier n'existe pas : {path}")
    try:
        df, meta = pyreadstat.read_dta(path)
        labels = meta.variable_value_labels
        return df, labels
    except Exception as e:
        raise ValueError(
            f"Impossible de lire le fichier Stata : {path}. Erreur : {e}"
        ) from e


def import_sas(path: str) -> tuple[pd.DataFrame, dict]:
    """Importe un fichier SAS (.sas7bdat) et retourne (df, labels).

    Args:
        path: Chemin vers le fichier .sas7bdat.

    Returns:
        Un tuple contenant :
        - df: Le DataFrame pandas avec les données.
        - labels: Un dictionnaire des labels de valeurs extraits via pyreadstat
          (meta.variable_value_labels).

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le fichier ne peut pas être lu ou n'est pas un fichier SAS valide.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Le fichier n'existe pas : {path}")
    try:
        df, meta = pyreadstat.read_sas7bdat(path)
        labels = meta.variable_value_labels
        return df, labels
    except Exception as e:
        raise ValueError(
            f"Impossible de lire le fichier SAS : {path}. Erreur : {e}"
        ) from e


def import_xml(path: str, xpath: str = "./*") -> pd.DataFrame:
    """Importe un fichier XML et retourne un DataFrame pandas.

    Nécessite la dépendance lxml (installée par défaut dans la Phase 15).

    Args:
        path: Chemin vers le fichier XML à importer.
        xpath: Expression XPath pour sélectionner les nœuds à convertir en lignes.
            Par défaut "./*" sélectionne tous les enfants directs du nœud racine.

    Returns:
        DataFrame pandas contenant les données du fichier XML.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le fichier ne peut pas être lu ou n'est pas un XML valide.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Le fichier n'existe pas : {path}")
    try:
        return pd.read_xml(path, xpath=xpath)
    except Exception as e:
        raise ValueError(f"Impossible de lire le fichier XML : {path}. Erreur : {e}") from e


def import_sql(connection_url: str, query: str) -> pd.DataFrame:
    """Importe des données depuis une base SQL et retourne un DataFrame pandas.

    Utilise SQLAlchemy pour la connexion. SQLite est supporté par défaut.
    Seules les requêtes SELECT sont autorisées (sécurité T15).

    Args:
        connection_url: URL de connexion SQLAlchemy (ex: "sqlite:///ma_base.db",
            "postgresql://user:pass@host/dbname").
        query: Requête SQL à exécuter. Doit être une requête SELECT.

    Returns:
        DataFrame pandas contenant les résultats de la requête.

    Raises:
        InvalidMethodInputError: Si la requête n'est pas un SELECT.
        ValueError: Si la connexion échoue ou si la requête ne peut pas être exécutée.

    Note:
        Cette fonction ne doit JAMAIS être appelée avec une requête fournie
        directement par un LLM (T15 hérité).
    """
    # Sécurité : n'autoriser que les requêtes SELECT
    query_stripped = query.strip().upper()
    if not query_stripped.startswith("SELECT"):
        raise InvalidMethodInputError(
            f"Seules les requêtes SELECT sont autorisées. "
            f"Requête reçue : '{query[:50]}...' "
            f"(tronquée). Utilisez une requête SELECT pour importer des données."
        )

    # Détection de requêtes dangereuses supplémentaires
    dangerous_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
    for keyword in dangerous_keywords:
        if keyword in query_stripped:
            raise InvalidMethodInputError(
                f"La requête contient le mot-clé interdit '{keyword}'. "
                f"Seules les requêtes SELECT sont autorisées pour l'import de données."
            )

    try:
        # Import paresseux de sqlalchemy (T18)
        from sqlalchemy import create_engine, text

        engine = create_engine(connection_url)
        # pd.read_sql gère nativement la conversion en DataFrame et évite 
        # les erreurs de typage mypy liées à result.fetchall() / result.keys()
        df = pd.read_sql(text(query), engine)
        return df
        
    except InvalidMethodInputError:
        # Re-raise les erreurs de validation telles quelles
        raise
    except Exception as e:
        raise ValueError(
            f"Impossible d'exécuter la requête SQL sur '{connection_url}'. Erreur : {e}"
        ) from e