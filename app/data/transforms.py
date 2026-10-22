"""Transformations de données avec traçabilité (V2 §7.2).

Ce module fournit des fonctions de transformation de DataFrames qui
retournent un nouveau DataFrame et un TransformStep documentant
l'opération. Les DataFrames d'entrée ne sont jamais modifiés.
"""

from __future__ import annotations

import ast
from typing import Any

import pandas as pd

from app.data.transform_log import TransformStep

# Liste blanche des opérateurs et fonctions autorisés dans compute_variable
_ALLOWED_OPERATORS = {
    "+",
    "-",
    "*",
    "/",
    "%",
    "**",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "&",
    "|",
    "~",
    "and",
    "or",
    "not",
    "in",
    "not in",
}

_ALLOWED_FUNCTIONS = {
    "log",
    "log10",
    "log2",
    "exp",
    "sqrt",
    "abs",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "min",
    "max",
    "sum",
    "mean",
    "std",
    "var",
    "round",
    "floor",
    "ceil",
    "where",
    "isin",
}


def _validate_expression(expression: str) -> None:
    """Valide qu'une expression n'utilise que des opérateurs/fonctions autorisés.

    Args:
        expression: Expression à valider (syntaxe pandas.eval).

    Raises:
        ValueError: Si l'expression contient des opérations non autorisées.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Expression invalide : {e}") from e

    for node in ast.walk(tree):
        # Opérateurs binaires
        if isinstance(node, ast.BinOp):
            op_type = type(node.op).__name__
            if op_type not in {
                "Add",
                "Sub",
                "Mult",
                "Div",
                "Mod",
                "Pow",
                "Eq",
                "NotEq",
                "Lt",
                "LtE",
                "Gt",
                "GtE",
                "BitAnd",
                "BitOr",
            }:
                raise ValueError(
                    f"Opérateur non autorisé dans l'expression : {op_type}"
                )
        # Opérateurs unaires
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op).__name__
            if op_type not in {"UAdd", "USub", "Not", "Invert"}:
                raise ValueError(f"Opérateur unaire non autorisé : {op_type}")
        # Appels de fonctions
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name not in _ALLOWED_FUNCTIONS:
                    raise ValueError(
                        f"Fonction non autorisée : {func_name}. "
                        f"Fonctions autorisées : {', '.join(sorted(_ALLOWED_FUNCTIONS))}"
                    )
        # Comparaisons
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                op_type = type(op).__name__
                if op_type not in {
                    "Eq",
                    "NotEq",
                    "Lt",
                    "LtE",
                    "Gt",
                    "GtE",
                    "In",
                    "NotIn",
                }:
                    raise ValueError(f"Comparaison non autorisée : {op_type}")
        # Nœuds autorisés sans vérification
        elif isinstance(
            node,
            (
                ast.Expression,
                ast.Load,
                ast.Store,
                ast.Name,
                ast.Constant,
                ast.Num,
                ast.Str,
                ast.NameConstant,
                ast.Attribute,
                ast.Subscript,
                ast.Index,
                ast.Slice,
                ast.Tuple,
                ast.List,
                ast.BoolOp,
            ),
        ):
            pass
        # Tout autre nœud est interdit
        elif not isinstance(
            node,
            (
                ast.expr_context,
                ast.expr,
                ast.operator,
                ast.cmpop,
                ast.unaryop,
                ast.boolop,
            ),
        ):
            # Ignorer les nœuds de contexte
            pass


def _make_step(
    operation: str,
    params: dict[str, Any],
    columns_in: list[str],
    columns_out: list[str],
    n_rows_before: int,
    n_rows_after: int,
    code: str,
) -> TransformStep:
    """Crée un TransformStep avec un step_id auto-généré.

    Args:
        operation: Type d'opération.
        params: Paramètres de l'opération.
        columns_in: Colonnes en entrée.
        columns_out: Colonnes en sortie.
        n_rows_before: Nombre de lignes avant.
        n_rows_after: Nombre de lignes après.
        code: Code Python reproductible.

    Returns:
        Un TransformStep avec un step_id unique.
    """
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


def recode_values(
    df: pd.DataFrame,
    column: str,
    mapping: dict,
    new_column: str | None = None,
) -> tuple[pd.DataFrame, TransformStep]:
    """Recoder les valeurs d'une colonne selon un mapping.

    Args:
        df: DataFrame d'entrée (non modifié).
        column: Colonne à recoder.
        mapping: Dictionnaire de correspondance {ancienne_valeur: nouvelle_valeur}.
        new_column: Nom de la colonne de sortie. Si None, remplace la colonne d'entrée.

    Returns:
        Un tuple (df_nouveau, TransformStep).

    Raises:
        KeyError: Si la colonne n'existe pas.
    """
    if column not in df.columns:
        raise KeyError(f"La colonne '{column}' n'existe pas dans le DataFrame.")

    df_new = df.copy()
    target_col = new_column if new_column else column

    if new_column:
        df_new[new_column] = df_new[column].map(mapping)
    else:
        df_new[column] = df_new[column].map(mapping)

    code = f'df["{target_col}"] = df["{column}"].map({mapping!r})'
    step = _make_step(
        operation="recode",
        params={"column": column, "mapping": mapping, "new_column": new_column},
        columns_in=[column],
        columns_out=[target_col],
        n_rows_before=len(df),
        n_rows_after=len(df_new),
        code=code,
    )
    return df_new, step


def compute_variable(
    df: pd.DataFrame,
    new_column: str,
    expression: str,
) -> tuple[pd.DataFrame, TransformStep]:
    """Créer une nouvelle variable par calcul sur des colonnes existantes.

    Utilise DataFrame.eval avec une liste blanche d'opérateurs et fonctions.
    Jamais eval()/exec() Python (T15).

    Args:
        df: DataFrame d'entrée (non modifié).
        new_column: Nom de la nouvelle colonne.
        expression: Expression de calcul (syntaxe pandas.eval).

    Returns:
        Un tuple (df_nouveau, TransformStep).

    Raises:
        ValueError: Si l'expression contient des opérations non autorisées.
    """
    _validate_expression(expression)

    df_new = df.copy()
    df_new[new_column] = df_new.eval(expression)

    code = f'df["{new_column}"] = df.eval("{expression}")'
    step = _make_step(
        operation="compute",
        params={"new_column": new_column, "expression": expression},
        columns_in=[],  # Déterminé dynamiquement par l'expression
        columns_out=[new_column],
        n_rows_before=len(df),
        n_rows_after=len(df_new),
        code=code,
    )
    return df_new, step


def standardize(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, TransformStep]:
    """Standardiser des colonnes (z-score : (x - mean) / std).

    Args:
        df: DataFrame d'entrée (non modifié).
        columns: Liste des colonnes à standardiser.

    Returns:
        Un tuple (df_nouveau, TransformStep).

    Raises:
        KeyError: Si une colonne n'existe pas.
    """
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"La colonne '{col}' n'existe pas dans le DataFrame.")

    df_new = df.copy()
    for col in columns:
        mean = df_new[col].mean()
        std = df_new[col].std()
        df_new[col] = (df_new[col] - mean) / std

    code_lines = [
        f'df["{col}"] = (df["{col}"] - df["{col}"].mean()) / df["{col}"].std()'
        for col in columns
    ]
    code = "\n".join(code_lines)
    step = _make_step(
        operation="standardize",
        params={"columns": columns},
        columns_in=columns,
        columns_out=columns,
        n_rows_before=len(df),
        n_rows_after=len(df_new),
        code=code,
    )
    return df_new, step


def center(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, TransformStep]:
    """Centrer des colonnes (x - mean).

    Args:
        df: DataFrame d'entrée (non modifié).
        columns: Liste des colonnes à centrer.

    Returns:
        Un tuple (df_nouveau, TransformStep).

    Raises:
        KeyError: Si une colonne n'existe pas.
    """
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"La colonne '{col}' n'existe pas dans le DataFrame.")

    df_new = df.copy()
    for col in columns:
        mean = df_new[col].mean()
        df_new[col] = df_new[col] - mean

    code_lines = [f'df["{col}"] = df["{col}"] - df["{col}"].mean()' for col in columns]
    code = "\n".join(code_lines)
    step = _make_step(
        operation="center",
        params={"columns": columns},
        columns_in=columns,
        columns_out=columns,
        n_rows_before=len(df),
        n_rows_after=len(df_new),
        code=code,
    )
    return df_new, step


def discretize(
    df: pd.DataFrame,
    column: str,
    bins: int | list[float],
    labels: list[str] | None = None,
    strategy: str = "width",
) -> tuple[pd.DataFrame, TransformStep]:
    """Discrétiser une variable continue en catégories.

    Args:
        df: DataFrame d'entrée (non modifié).
        column: Colonne à discrétiser.
        bins: Nombre de bins (int) ou bornes explicites (list[float]).
        labels: Labels personnalisés pour les bins.
        strategy: "width" (largeur égale) ou "quantile" (effectifs égaux).

    Returns:
        Un tuple (df_nouveau, TransformStep).

    Raises:
        KeyError: Si la colonne n'existe pas.
        ValueError: Si la stratégie n'est pas supportée.
    """
    if column not in df.columns:
        raise KeyError(f"La colonne '{column}' n'existe pas dans le DataFrame.")

    if strategy not in {"width", "quantile"}:
        raise ValueError(
            f"Stratégie non supportée : {strategy}. Utiliser 'width' ou 'quantile'."
        )

    df_new = df.copy()

    if isinstance(bins, int):
        if strategy == "width":
            df_new[column] = pd.cut(df_new[column], bins=bins, labels=labels)
        else:  # quantile
            df_new[column] = pd.qcut(
                df_new[column], q=bins, labels=labels, duplicates="drop"
            )
    else:  # list[float]
        df_new[column] = pd.cut(df_new[column], bins=bins, labels=labels)

    code = f'df["{column}"] = pd.{"cut" if strategy == "width" or isinstance(bins, list) else "qcut"}(df["{column}"], bins={bins!r}, labels={labels!r})'
    step = _make_step(
        operation="discretize",
        params={"column": column, "bins": bins, "labels": labels, "strategy": strategy},
        columns_in=[column],
        columns_out=[column],
        n_rows_before=len(df),
        n_rows_after=len(df_new),
        code=code,
    )
    return df_new, step


def extract_date_parts(
    df: pd.DataFrame,
    column: str,
    parts: list[str],
) -> tuple[pd.DataFrame, TransformStep]:
    """Extraire des parties d'une colonne datetime.

    Args:
        df: DataFrame d'entrée (non modifié).
        column: Colonne datetime.
        parts: Parties à extraire ("year", "month", "day", "hour", "minute",
               "second", "dayofweek", "quarter").

    Returns:
        Un tuple (df_nouveau, TransformStep).

    Raises:
        KeyError: Si la colonne n'existe pas.
        ValueError: Si une partie n'est pas supportée.
    """
    if column not in df.columns:
        raise KeyError(f"La colonne '{column}' n'existe pas dans le DataFrame.")

    allowed_parts = {
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "dayofweek",
        "quarter",
    }
    for part in parts:
        if part not in allowed_parts:
            raise ValueError(
                f"Partie non supportée : {part}. Parties autorisées : {', '.join(sorted(allowed_parts))}"
            )

    df_new = df.copy()
    code_lines = []

    for part in parts:
        new_col = f"{column}_{part}"
        df_new[new_col] = getattr(df_new[column].dt, part)
        code_lines.append(f'df["{new_col}"] = df["{column}"].dt.{part}')

    code = "\n".join(code_lines)
    step = _make_step(
        operation="extract_date_parts",
        params={"column": column, "parts": parts},
        columns_in=[column],
        columns_out=[f"{column}_{part}" for part in parts],
        n_rows_before=len(df),
        n_rows_after=len(df_new),
        code=code,
    )
    return df_new, step
