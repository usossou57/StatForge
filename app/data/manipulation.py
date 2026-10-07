"""Manipulation de DataFrames avec traçabilité (V2 T15.8).

Ce module fournit des fonctions de manipulation de DataFrames (fusion,
agrégation, pivot, dédoublonnage, échantillonnage) qui retournent un
nouveau DataFrame et un TransformStep documentant l'opération.
Les DataFrames d'entrée ne sont jamais modifiés.
"""

from __future__ import annotations

import time
from typing import Any, Literal

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


def merge_datasets(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    on: str | list[str],
    how: Literal["inner", "left", "right", "outer"] = "inner",
    suffixes: tuple[str, str] = ("_x", "_y"),
) -> tuple[pd.DataFrame, TransformStep]:
    """Fusionne deux DataFrames avec détection de clés dupliquées.

    Args:
        df_left: DataFrame de gauche (non modifié).
        df_right: DataFrame de droite (non modifié).
        on: Colonne(s) de jointure.
        how: Type de jointure ('inner', 'left', 'right', 'outer').
        suffixes: Suffixes pour les colonnes en cas de doublon.

    Returns:
        Un tuple (df_fusionné, TransformStep).

    Raises:
        InvalidMethodInputError: Si le type de jointure est invalide ou
            si des clés dupliquées sont détectées dans les DataFrames d'entrée.
    """
    valid_how = {"inner", "left", "right", "outer"}
    if how not in valid_how:
        raise InvalidMethodInputError(
            f"Type de jointure invalide : '{how}'. "
            f"Types valides : {', '.join(sorted(valid_how))}."
        )

    on_list = [on] if isinstance(on, str) else on

    # Détection de clés dupliquées
    left_dupes = int(df_left.duplicated(subset=on_list).sum())
    right_dupes = int(df_right.duplicated(subset=on_list).sum())
    warnings_list: list[str] = []
    if left_dupes > 0:
        warnings_list.append(
            f"Attention : {left_dupes} clé(s) dupliquée(s) dans le "
            f"DataFrame de gauche sur {on_list}. Cela peut produire "
            f"un produit cartésien partiel."
        )
    if right_dupes > 0:
        warnings_list.append(
            f"Attention : {right_dupes} clé(s) dupliquée(s) dans le "
            f"DataFrame de droite sur {on_list}. Cela peut produire "
            f"un produit cartésien partiel."
        )

    df_merged = pd.merge(
        df_left, df_right, on=on_list, how=how, suffixes=suffixes
    )

    on_str = repr(on)
    code = (
        f"df = pd.merge(df_left, df_right, on={on_str}, "
        f"how='{how}', suffixes={suffixes!r})"
    )
    if warnings_list:
        code += f"\n# AVERTISSEMENTS : {' | '.join(warnings_list)}"

    step = _make_step(
        operation="merge",
        params={
            "on": on_list,
            "how": how,
            "suffixes": list(suffixes),
            "left_dupes": left_dupes,
            "right_dupes": right_dupes,
            "warnings": warnings_list,
        },
        columns_in=list(df_left.columns) + list(df_right.columns),
        columns_out=list(df_merged.columns),
        n_rows_before=len(df_left) + len(df_right),
        n_rows_after=len(df_merged),
        code=code,
    )
    return df_merged, step


def aggregate_by(
    df: pd.DataFrame,
    group_cols: str | list[str],
    agg_dict: dict[str, str | list[str]],
) -> tuple[pd.DataFrame, TransformStep]:
    """Agrège un DataFrame par groupe.

    Args:
        df: DataFrame d'entrée (non modifié).
        group_cols: Colonne(s) de groupement.
        agg_dict: Dictionnaire d'agrégation.
            Ex: {"score": "mean", "age": ["min", "max"]}

    Returns:
        Un tuple (df_agrégé, TransformStep).

    Raises:
        KeyError: Si une colonne n'existe pas.
    """
    group_list = [group_cols] if isinstance(group_cols, str) else group_cols

    for col in group_list:
        if col not in df.columns:
            raise KeyError(f"La colonne '{col}' n'existe pas dans le DataFrame.")
    for col in agg_dict:
        if col not in df.columns:
            raise KeyError(f"La colonne '{col}' n'existe pas dans le DataFrame.")

    df_agg = df.groupby(group_list, as_index=False).agg(agg_dict)
    # Aplatir les colonnes MultiIndex si nécessaire
    if isinstance(df_agg.columns, pd.MultiIndex):
        df_agg.columns = [
            f"{col}_{agg}" if agg else col
            for col, agg in df_agg.columns
        ]

    code = (
        f"df = df.groupby({group_list!r}, as_index=False)"
        f".agg({agg_dict!r})"
    )

    step = _make_step(
        operation="aggregate",
        params={"group_cols": group_list, "agg_dict": agg_dict},
        columns_in=list(df.columns),
        columns_out=list(df_agg.columns),
        n_rows_before=len(df),
        n_rows_after=len(df_agg),
        code=code,
    )
    return df_agg, step


def pivot_longer(
    df: pd.DataFrame,
    id_vars: list[str],
    value_vars: list[str],
    var_name: str = "variable",
    value_name: str = "value",
) -> tuple[pd.DataFrame, TransformStep]:
    """Transforme un DataFrame du format large au format long (melt).

    Args:
        df: DataFrame d'entrée (non modifié).
        id_vars: Colonnes à conserver comme identifiants.
        value_vars: Colonnes à transformer en format long.
        var_name: Nom de la colonne des variables.
        value_name: Nom de la colonne des valeurs.

    Returns:
        Un tuple (df_long, TransformStep).
    """
    df_long = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
    )

    code = (
        f"df = pd.melt(df, id_vars={id_vars!r}, "
        f"value_vars={value_vars!r}, "
        f"var_name='{var_name}', value_name='{value_name}')"
    )

    step = _make_step(
        operation="pivot_longer",
        params={
            "id_vars": id_vars,
            "value_vars": value_vars,
            "var_name": var_name,
            "value_name": value_name,
        },
        columns_in=list(df.columns),
        columns_out=list(df_long.columns),
        n_rows_before=len(df),
        n_rows_after=len(df_long),
        code=code,
    )
    return df_long, step


def pivot_wider(
    df: pd.DataFrame,
    index: str | list[str],
    columns: str,
    values: str,
    aggfunc: Literal["first", "mean", "sum", "min", "max", "count", "std", "var"] = "first",
) -> tuple[pd.DataFrame, TransformStep]:
    """Transforme un DataFrame du format long au format large (pivot).

    Args:
        df: DataFrame d'entrée (non modifié).
        index: Colonne(s) à utiliser comme index.
        columns: Colonne dont les valeurs deviennent les nouvelles colonnes.
        values: Colonne dont les valeurs remplissent le tableau.
        aggfunc: Fonction d'agrégation en cas de doublons ('first', 'mean', etc.).

    Returns:
        Un tuple (df_wide, TransformStep).
    """
    df_wide = pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
    ).reset_index()
    # Aplatir les colonnes MultiIndex
    if isinstance(df_wide.columns, pd.MultiIndex):
        df_wide.columns = [
            f"{col}_{val}" if val else col
            for col, val in df_wide.columns
        ]

    code = (
        f"df = pd.pivot_table(df, index={index!r}, "
        f"columns='{columns}', values='{values}', "
        f"aggfunc='{aggfunc}').reset_index()"
    )

    step = _make_step(
        operation="pivot_wider",
        params={
            "index": index,
            "columns": columns,
            "values": values,
            "aggfunc": aggfunc,
        },
        columns_in=list(df.columns),
        columns_out=list(df_wide.columns),
        n_rows_before=len(df),
        n_rows_after=len(df_wide),
        code=code,
    )
    return df_wide, step


def transpose_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, TransformStep]:
    """Transpose un DataFrame (lignes ↔ colonnes).

    Args:
        df: DataFrame d'entrée (non modifié).

    Returns:
        Un tuple (df_transposé, TransformStep).
    """
    df_t = df.T.reset_index()
    df_t.columns = [
        f"col_{i}" if i > 0 else "index"
        for i in range(len(df_t.columns))
    ]

    code = "df = df.T.reset_index()"

    step = _make_step(
        operation="transpose",
        params={},
        columns_in=list(df.columns),
        columns_out=list(df_t.columns),
        n_rows_before=len(df),
        n_rows_after=len(df_t),
        code=code,
    )
    return df_t, step


def drop_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: Literal["first", "last", False] = "first",
) -> tuple[pd.DataFrame, TransformStep]:
    """Supprime les lignes dupliquées.

    Args:
        df: DataFrame d'entrée (non modifié).
        subset: Colonnes à considérer pour la détection des doublons.
            Si None, toutes les colonnes sont utilisées.
        keep: Quel doublon conserver ('first', 'last', False pour tous supprimer).

    Returns:
        Un tuple (df_dédoublonné, TransformStep).

    Raises:
        InvalidMethodInputError: Si la valeur de 'keep' est invalide.
    """
    valid_keep ={"first", "last", False}
    if keep not in valid_keep:
        raise InvalidMethodInputError(
            f"Valeur de 'keep' invalide : {keep!r}. "
            f"Valeurs valides : 'first', 'last', False."
        )

    df_dedup = df.drop_duplicates(subset=subset, keep=keep)
    n_removed = len(df) - len(df_dedup)

    code = f"df = df.drop_duplicates(subset={subset!r}, keep={keep!r})"
    code += f"\n# {n_removed} ligne(s) dupliquée(s) supprimée(s)"

    step = _make_step(
        operation="drop_duplicates",
        params={
            "subset": subset,
            "keep": str(keep),
            "n_removed": n_removed,
        },
        columns_in=list(df.columns),
        columns_out=list(df_dedup.columns),
        n_rows_before=len(df),
        n_rows_after=len(df_dedup),
        code=code,
    )
    return df_dedup, step


def sample_rows(
    df: pd.DataFrame,
    n: int | None = None,
    frac: float | None = None,
    random_state: int = 0,
    replace: bool = False,
) -> tuple[pd.DataFrame, TransformStep]:
    """Échantillonne des lignes du DataFrame.

    La graine (random_state) est obligatoire pour la reproductibilité.

    Args:
        df: DataFrame d'entrée (non modifié).
        n: Nombre de lignes à échantillonner.
        frac: Fraction de lignes à échantillonner (0.0 à 1.0).
        random_state: Graine aléatoire (obligatoire, défaut 0).
        replace: Si True, échantillonnage avec remise.

    Returns:
        Un tuple (df_échantillon, TransformStep).

    Raises:
        InvalidMethodInputError: Si ni n ni frac n'est fourni, ou si les
            deux sont fournis simultanément.
    """
    if n is None and frac is None:
        raise InvalidMethodInputError(
            "Vous devez fournir soit 'n' (nombre de lignes), "
            "soit 'frac' (fraction), mais pas les deux."
        )
    if n is not None and frac is not None:
        raise InvalidMethodInputError(
            "Fournissez soit 'n', soit 'frac', pas les deux simultanément."
        )

    df_sample = df.sample(
        n=n, frac=frac, random_state=random_state, replace=replace
    )

    if n is not None:
        code = (
            f"df = df.sample(n={n}, random_state={random_state}, "
            f"replace={replace})"
        )
    else:
        code = (
            f"df = df.sample(frac={frac}, random_state={random_state}, "
            f"replace={replace})"
        )

    step = _make_step(
        operation="sample",
        params={
            "n": n,
            "frac": frac,
            "random_state": random_state,
            "replace": replace,
        },
        columns_in=list(df.columns),
        columns_out=list(df_sample.columns),
        n_rows_before=len(df),
        n_rows_after=len(df_sample),
        code=code,
    )
    return df_sample, step
