"""Module de profilage des datasets.

Ce module fournit des fonctions pour analyser et profiler les données
brutes importées, afin de construire des dictionnaires de données complets.
"""

import hashlib
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from app.data.models import DataDictionary, Variable


def profile_dataset(df: pd.DataFrame) -> dict:
    """Calcule un profil descriptif complet d'un DataFrame.

    Args:
        df: Le DataFrame à profiler.

    Returns:
        Un dictionnaire contenant :
        - n_rows: nombre de lignes
        - n_columns: nombre de colonnes
        - dtypes: dict colonne -> type pandas
        - missing_pct: dict colonne -> pourcentage de valeurs manquantes
        - n_unique: dict colonne -> cardinalité
        - constant_columns: liste des colonnes à valeur unique
    """
    n_rows, n_columns = df.shape
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_pct = (df.isna().sum() / len(df) * 100).to_dict()
    n_unique = df.nunique().to_dict()
    constant_columns = [col for col in df.columns if df[col].nunique() <= 1]

    return {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "dtypes": dtypes,
        "missing_pct": missing_pct,
        "n_unique": n_unique,
        "constant_columns": constant_columns,
    }


def describe_variable(df: pd.DataFrame, column: str) -> dict:
    """Calcule les statistiques descriptives d'une variable.

    Args:
        df: Le DataFrame contenant la variable.
        column: Nom de la colonne à décrire.

    Returns:
        Pour une variable numérique : dict avec mean, median, std, min, max, n_missing.
        Pour une variable catégorielle : dict avec frequencies (dict modalité -> effectif).

    Raises:
        KeyError: Si la colonne n'existe pas dans le DataFrame.
    """
    if column not in df.columns:
        raise KeyError(f"La colonne '{column}' n'existe pas dans le DataFrame.")

    if is_numeric_dtype(df[column]):
        return {
            "mean": float(df[column].mean()),
            "median": float(df[column].median()),
            "std": float(df[column].std()),
            "min": float(df[column].min()),
            "max": float(df[column].max()),
            "n_missing": int(df[column].isna().sum()),
        }
    else:
        frequencies = df[column].value_counts().to_dict()
        return {"frequencies": frequencies}


def build_data_dictionary(
    df: pd.DataFrame, dataset_name: str, source_path: str
) -> DataDictionary:
    """Construit un dictionnaire de données complet à partir d'un DataFrame."""
    n_rows, n_columns = df.shape
    variables = {}

    for col in df.columns:
        # Déterminer le type de données
        if is_numeric_dtype(df[col]):
            dtype = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            dtype = "datetime"
        elif (
            isinstance(df[col].dtype, pd.CategoricalDtype)
            or df[col].dtype == object
            or pd.api.types.is_string_dtype(df[col])
        ):
            dtype = "categorical"
        else:
            dtype = "unknown"

        # Déterminer le type sémantique
        semantic_type = None
        if dtype == "numeric":
            n_unique = df[col].nunique()
            is_integer = (
                df[col].dropna().apply(lambda x: x == int(x)).all()
                if n_unique > 0
                else False
            )

            if n_unique == 2:
                semantic_type = "binary"
            elif is_integer:
                semantic_type = "count"
            elif n_unique > 10:
                semantic_type = "continuous"
        elif dtype == "categorical":
            n_unique = df[col].nunique()
            if n_unique == 2:
                semantic_type = "binary"

        # Niveaux pour variables catégorielles
        levels = None
        if dtype == "categorical":
            levels = [str(val) for val in df[col].dropna().unique()]

        variables[col] = Variable(
            name=col,
            dtype=dtype,
            semantic_type=semantic_type,
            levels=levels,
            missing_values=[],
        )

    file_hash = _compute_file_hash(source_path)

    return DataDictionary(
        dataset_name=dataset_name,
        n_rows=n_rows,
        n_columns=n_columns,
        variables=variables,
        source_path=source_path,
        file_hash=file_hash,
    )


def _compute_file_hash(file_path: str) -> str:
    """Calcule le hash SHA256 d'un fichier."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Le fichier n'existe pas : {file_path}")

    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
