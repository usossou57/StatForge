"""Gestion des valeurs manquantes (V2 §7.2 / T15.7).

Ce module fournit des fonctions pour détecter, recoder et imputer
les valeurs manquantes dans un DataFrame pandas. Chaque opération
de modification retourne un nouveau DataFrame et un TransformStep
pour assurer la traçabilité.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.data.transform_log import TransformStep
from app.errors import InvalidMethodInputError


def _make_step(
    operation: str,
    params: dict[str, Any],
    columns_in: list[str],
    columns_out: list[str],
    n_rows_before: int,
    n_rows_after: int,
    code: str,
) -> TransformStep:
    """Crée un TransformStep avec un step_id auto-généré."""
    import time
    step_id = f"step_{int(time.time() * 1000)}"
    return TransformStep(
        step_id=step_id,
        operation=operation,
        params=params,
        columns_in=columns_in,
        columns_out=columns_out,
        n_rows_before=n_rows_before,
        n_rows_after=n_rows_after,
        code=code,
    )


def detect_missing(df: pd.DataFrame) -> dict[str, Any]:
    """Détecte et résume les valeurs manquantes dans un DataFrame.

    Args:
        df: DataFrame à analyser.

    Returns:
        Un dictionnaire contenant :
        - 'total_missing': Nombre total de valeurs manquantes.
        - 'pct_missing': Pourcentage total de valeurs manquantes.
        - 'by_column': Dictionnaire avec le taux de manquants par colonne.
        - 'patterns': Liste des motifs de manquants les plus fréquents
          (sous forme de tuples (motif, count, pct)).
    """
    total_cells = df.size
    total_missing = int(df.isna().sum().sum())
    pct_missing = (total_missing / total_cells * 100) if total_cells > 0 else 0.0

    by_column = {}
    for col in df.columns:
        col_missing = int(df[col].isna().sum())
        by_column[col] = {
            "count": col_missing,
            "pct": round(col_missing / len(df) * 100, 2) if len(df) > 0 else 0.0,
        }

    # Motifs de manquants (lignes avec au moins un manquant)
    missing_mask = df.isna()
    if missing_mask.any().any():
        # Créer une signature de motif pour chaque ligne (ex: "A:True,B:False")
        patterns = missing_mask.apply(
            lambda row: ",".join([f"{col}:{val}" for col, val in row.items() if val]),
            axis=1,
        )
        pattern_counts = patterns[patterns != ""].value_counts()
        top_patterns = [
            (pattern, count, round(count / len(patterns[patterns != ""]) * 100, 2))
            for pattern, count in pattern_counts.head(5).items()
        ]
    else:
        top_patterns = []

    return {
        "total_missing": total_missing,
        "pct_missing": round(pct_missing, 2),
        "by_column": by_column,
        "patterns": top_patterns,
    }


def recode_missing(
    df: pd.DataFrame,
    column: str,
    codes: list[Any],
) -> tuple[pd.DataFrame, TransformStep]:
    """Recoder des valeurs spécifiques (ex: -99, 999) en valeurs manquantes (NaN).

    Args:
        df: DataFrame d'entrée (non modifié).
        column: Colonne à traiter.
        codes: Liste des valeurs à considérer comme manquantes.

    Returns:
        Un tuple (df_nouveau, TransformStep).

    Raises:
        KeyError: Si la colonne n'existe pas.
    """
    if column not in df.columns:
        raise KeyError(f"La colonne '{column}' n'existe pas dans le DataFrame.")

    df_new = df.copy()
    # Remplacer les codes par pd.NA (ou np.nan pour la compatibilité)
    df_new[column] = df_new[column].replace(codes, pd.NA)

    codes_str = ", ".join(repr(c) for c in codes)
    code = f'df["{column}"] = df["{column}"].replace([{codes_str}], pd.NA)'
    
    step = _make_step(
        operation="recode_missing",
        params={"column": column, "codes": codes},
        columns_in=[column],
        columns_out=[column],
        n_rows_before=len(df),
        n_rows_after=len(df_new),
        code=code,
    )
    return df_new, step


def impute_simple(
    df: pd.DataFrame,
    column: str,
    strategy: str,
    value: Any | None = None,
) -> tuple[pd.DataFrame, TransformStep]:
    """Imputer les valeurs manquantes avec une stratégie simple.

    Args:
        df: DataFrame d'entrée (non modifié).
        column: Colonne à imputer.
        strategy: Stratégie d'imputation : 'mean', 'median', 'mode', ou 'constant'.
        value: Valeur à utiliser si strategy == 'constant'.

    Returns:
        Un tuple (df_nouveau, TransformStep).

    Raises:
        KeyError: Si la colonne n'existe pas.
        InvalidMethodInputError: Si la stratégie est invalide ou si 'value' est manquant.
    """
    if column not in df.columns:
        raise KeyError(f"La colonne '{column}' n'existe pas dans le DataFrame.")

    valid_strategies = {"mean", "median", "mode", "constant"}
    if strategy not in valid_strategies:
        raise InvalidMethodInputError(
            f"Stratégie d'imputation invalide : '{strategy}'. "
            f"Stratégies valides : {', '.join(sorted(valid_strategies))}."
        )

    df_new = df.copy()

    if strategy == "mean":
        fill_val = float(df_new[column].mean())
        df_new[column] = df_new[column].fillna(fill_val)
        code = f'df["{column}"] = df["{column}"].fillna({fill_val:.4f})  # mean'
    elif strategy == "median":
        fill_val = float(df_new[column].median())
        df_new[column] = df_new[column].fillna(fill_val)
        code = f'df["{column}"] = df["{column}"].fillna({fill_val:.4f})  # median'
    elif strategy == "mode":
        # mode() retourne une Series, on prend la première valeur
        mode_val = df_new[column].mode().iloc[0]
        fill_val = mode_val
        df_new[column] = df_new[column].fillna(fill_val)
        code = f'df["{column}"] = df["{column}"].fillna({fill_val!r})  # mode'
    elif strategy == "constant":
        if value is None:
            raise InvalidMethodInputError(
                "La stratégie 'constant' nécessite le paramètre 'value'."
            )
        fill_val = value
        df_new[column] = df_new[column].fillna(fill_val)
        code = f'df["{column}"] = df["{column}"].fillna({fill_val!r})  # constant'
    else:
        # Ne devrait pas arriver grâce à la validation ci-dessus
        raise InvalidMethodInputError(f"Stratégie non implémentée : {strategy}")

    step = _make_step(
        operation="impute_simple",
        params={"column": column, "strategy": strategy, "value": value},
        columns_in=[column],
        columns_out=[column],
        n_rows_before=len(df),
        n_rows_after=len(df_new),
        code=code,
    )
    return df_new, step
