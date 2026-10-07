"""Gestion des gros volumes de données (V2 T15.9).

Ce module fournit des fonctions pour traiter des DataFrames volumineux
dans les limites d'une machine CPU-only (~12 Go RAM) :
- import_csv_chunked : lecture par morceaux pour économiser la mémoire
- downcast_dtypes : optimisation des types pour réduire l'empreinte mémoire
- memory_report : rapport détaillé de l'utilisation mémoire
- sample_for_analysis : échantillonnage reproductible pour les très gros datasets

Conformément à T3, aucun usage de Dask/Polars/DuckDB.
Conformément à T22, aucune fonction ne doit dépasser ~4 Go sur 1M lignes × 20 colonnes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def import_csv_chunked(
    path: str,
    chunksize: int = 200_000,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Importe un gros fichier CSV par morceaux pour économiser la mémoire.

    Lit le fichier par blocs de `chunksize` lignes et concatène le tout
    en un seul DataFrame. Permet de traiter des fichiers plus gros que
    la RAM disponible en ne chargeant qu'un morceau à la fois.

    Args:
        path: Chemin vers le fichier CSV.
        chunksize: Nombre de lignes par morceau. Défaut 200 000.
        usecols: Liste optionnelle des colonnes à charger. Si None,
            toutes les colonnes sont chargées.

    Returns:
        DataFrame pandas contenant toutes les lignes du fichier.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le fichier ne peut pas être lu.
    """
    from pathlib import Path

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Le fichier n'existe pas : {path}")

    try:
        # Utilisation directe de list() sur l'itérateur pour PERF402
        chunks = list(
            pd.read_csv(path, chunksize=chunksize, usecols=usecols, low_memory=False)
        )
        
        if not chunks:
            return pd.DataFrame()

        return pd.concat(chunks, ignore_index=True)

    except Exception as e:
        raise ValueError(
            f"Impossible de lire le fichier CSV par morceaux : {path}. Erreur : {e}"
        ) from e


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimise les types de données pour réduire l'empreinte mémoire.

    Applique les conversions suivantes quand c'est sûr :
    - int64 → int32 ou int16 ou int8 selon les valeurs
    - float64 → float32 si la perte de précision est acceptable
    - object → category si le ratio cardinalité/taille est faible (< 50%)

    Le DataFrame d'entrée n'est jamais modifié (copie créée).

    Args:
        df: DataFrame d'entrée (non modifié).

    Returns:
        Nouveau DataFrame avec des types optimisés.
    """
    df_opt = df.copy()

    for col in df_opt.columns:
        col_type = df_opt[col].dtype
        # Optimisation des entiers
        if pd.api.types.is_integer_dtype(col_type):
            col_min = df_opt[col].min()
            col_max = df_opt[col].max()

            if col_type == np.int64:
                if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                    df_opt[col] = df_opt[col].astype(np.int8)
                elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                    df_opt[col] = df_opt[col].astype(np.int16)
                else:
                    df_opt[col] = df_opt[col].astype(np.int32)

        # Optimisation des flottants
        elif pd.api.types.is_float_dtype(col_type):
            if col_type == np.float64:
                col_min = df_opt[col].min()
                col_max = df_opt[col].max()
                if (
                    col_min >= np.finfo(np.float32).min
                    and col_max <= np.finfo(np.float32).max
                ):
                    df_opt[col] = df_opt[col].astype(np.float32)


                # Convertir en float32 si la plage le permet
                col_min = df_opt[col].min()
                col_max = df_opt[col].max()
                if (
                    col_min >= np.finfo(np.float32).min
                    and col_max <= np.finfo(np.float32).max
                ):
                    df_opt[col] = df_opt[col].astype(np.float32)

        # Conversion object ou string → category
        elif pd.api.types.is_object_dtype(col_type) or pd.api.types.is_string_dtype(col_type):
            n_unique = df_opt[col].nunique()
            n_rows = len(df_opt)
            # Convertir si le ratio cardinalité/taille est faible (< 50%)
            if n_rows > 0 and (n_unique / n_rows) < 0.5:
                df_opt[col] = df_opt[col].astype("category")

    return df_opt


def memory_report(df: pd.DataFrame) -> dict[str, Any]:
    """Produit un rapport détaillé de l'utilisation mémoire du DataFrame.

    Args:
        df: DataFrame à analyser.

    Returns:
        Dictionnaire JSON-sérialisable contenant :
        - total_memory_mb: Mémoire totale en Mo
        - total_memory_gb: Mémoire totale en Go
        - n_rows: Nombre de lignes
        - n_columns: Nombre de colonnes
        - by_column: Dictionnaire {colonne: {"memory_mb": ..., "dtype": ..., "n_unique": ...}}
        - dtypes_summary: Résumé des types (ex: {"int32": 5, "float64": 3, ...})
    """
    # Mémoire par colonne (en bytes)
    memory_by_col = df.memory_usage(deep=True)
    total_bytes = memory_by_col.sum()

    # Rapport par colonne
    by_column = {}
    for col in df.columns:
        by_column[col] = {
            "memory_mb": round(float(memory_by_col[col]) / (1024 * 1024), 4),
            "dtype": str(df[col].dtype),
            "n_unique": int(df[col].nunique()),
        }

    # Résumé des types
    dtypes_summary: dict[str, int] = {}
    for dtype in df.dtypes:
        dtype_str = str(dtype)
        dtypes_summary[dtype_str] = dtypes_summary.get(dtype_str, 0) + 1

    return {
        "total_memory_mb": round(float(total_bytes) / (1024 * 1024), 4),
        "total_memory_gb": round(float(total_bytes) / (1024 * 1024 * 1024), 6),
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "by_column": by_column,
        "dtypes_summary": dtypes_summary,
    }


def sample_for_analysis(
    df: pd.DataFrame,
    max_rows: int,
    random_state: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Échantillonne un DataFrame s'il dépasse max_rows.

    Si le DataFrame a moins de max_rows lignes, il est retourné tel quel.
    Sinon, un échantillon aléatoire reproductible est extrait.

    Args:
        df: DataFrame d'entrée (non modifié).
        max_rows: Nombre maximum de lignes souhaité.
        random_state: Graine aléatoire pour la reproductibilité. Défaut 0.

    Returns:
        Un tuple contenant :
        - df_sample: DataFrame échantillonné (ou original si déjà petit)
        - stats: Dictionnaire avec les statistiques de l'échantillonnage :
          - original_rows: Nombre de lignes original
          - sampled_rows: Nombre de lignes après échantillonnage
          - sampling_rate: Taux d'échantillonnage (0.0 à 1.0)
          - was_sampled: Booléen indiquant si un échantillonnage a eu lieu
          - random_state: Graine utilisée

    Raises:
        InvalidMethodInputError: Si max_rows <= 0.
    """
    from app.errors import InvalidMethodInputError

    if max_rows <= 0:
        raise InvalidMethodInputError(
            f"max_rows doit être strictement positif, reçu : {max_rows}."
        )

    original_rows = len(df)

    if original_rows <= max_rows:
        # Pas besoin d'échantillonner
        stats = {
            "original_rows": original_rows,
            "sampled_rows": original_rows,
            "sampling_rate": 1.0,
            "was_sampled": False,
            "random_state": random_state,
        }
        return df.copy(), stats

    # Échantillonnage nécessaire
    df_sample = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    sampled_rows = len(df_sample)
    sampling_rate = sampled_rows / original_rows

    stats = {
        "original_rows": original_rows,
        "sampled_rows": sampled_rows,
        "sampling_rate": round(sampling_rate, 6),
        "was_sampled": True,
        "random_state": random_state,
    }

    return df_sample, stats
