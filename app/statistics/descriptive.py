"""Statistiques descriptives : tableaux croisés et fonctions associées."""

import pandas as pd


def _is_categorical(series: pd.Series) -> bool:
    """Vérifie si une série pandas doit être traitée comme catégorielle."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        return True
    return bool(pd.api.types.is_numeric_dtype(series) and series.nunique() <= 10)


def create_contingency_table(df: pd.DataFrame, row: str, col: str) -> pd.DataFrame:
    """Calcule un tableau croisé entre deux variables catégorielles."""
    if row not in df.columns:
        raise ValueError(f"La colonne '{row}' n'existe pas dans le DataFrame.")
    if col not in df.columns:
        raise ValueError(f"La colonne '{col}' n'existe pas dans le DataFrame.")
    
    if not _is_categorical(df[row]):
        raise ValueError(f"La variable '{row}' n'est pas catégorielle.")
    if not _is_categorical(df[col]):
        raise ValueError(f"La variable '{col}' n'est pas catégorielle.")
    
    return pd.crosstab(index=df[row], columns=df[col])
