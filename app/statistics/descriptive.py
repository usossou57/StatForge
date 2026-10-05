"""Statistiques descriptives et tableaux croisés.

Fonctions utilitaires pour l'exploration des données avant analyse.
"""

from __future__ import annotations

import pandas as pd


def create_contingency_table(
    df: pd.DataFrame, row: str, col: str
) -> pd.DataFrame:
    """Crée un tableau croisé (contingence) entre deux variables catégorielles.

    Args:
        df: DataFrame contenant les données.
        row: Nom de la variable pour les lignes.
        col: Nom de la variable pour les colonnes.

    Returns:
        DataFrame pivoté avec les effectifs croisés.

    Raises:
        ValueError: Si une des variables n'existe pas dans le DataFrame.
    """
    if row not in df.columns:
        raise ValueError(f"La variable '{row}' n'existe pas dans le DataFrame.")
    if col not in df.columns:
        raise ValueError(f"La variable '{col}' n'existe pas dans le DataFrame.")

    return pd.crosstab(df[row], df[col], margins=True, margins_name="Total")
