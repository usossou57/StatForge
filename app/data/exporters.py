"""Module d'export de DataFrames pandas vers différents formats.

Ce module fournit des fonctions pour exporter des DataFrames pandas
vers des formats courants (CSV, Excel, Parquet, Stata, LaTeX, HTML).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadstat


def _ensure_dir(path: str) -> None:
    """Vérifie et crée le répertoire parent du fichier s'il n'existe pas."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)


def export_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """Exporte un DataFrame vers un fichier CSV.

    Args:
        df: Le DataFrame à exporter.
        path: Chemin de destination du fichier CSV.
        index: Si True, écrit les index des lignes. Par défaut False.
    """
    _ensure_dir(path)
    df.to_csv(path, index=index, encoding="utf-8")


def export_xlsx(
    df: pd.DataFrame, path: str, sheet_name: str = "Feuil1", index: bool = False
) -> None:
    """Exporte un DataFrame vers un fichier Excel (.xlsx).

    Args:
        df: Le DataFrame à exporter.
        path: Chemin de destination du fichier Excel.
        sheet_name: Nom de la feuille de calcul. Par défaut "Feuil1".
        index: Si True, écrit les index des lignes. Par défaut False.
    """
    _ensure_dir(path)
    df.to_excel(path, sheet_name=sheet_name, index=index, engine="openpyxl")


def export_parquet(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """Exporte un DataFrame vers un fichier Parquet.

    Args:
        df: Le DataFrame à exporter.
        path: Chemin de destination du fichier Parquet.
        index: Si True, écrit les index des lignes. Par défaut False.
    """
    _ensure_dir(path)
    df.to_parquet(path, index=index, engine="pyarrow")


def export_stata(
    df: pd.DataFrame,
    path: str,
    labels: dict[str, dict[int | float, str]] | None = None,
) -> None:
    """Exporte un DataFrame vers un fichier Stata (.dta).

    Args:
        df: Le DataFrame à exporter.
        path: Chemin de destination du fichier Stata.
        labels: Dictionnaire optionnel des labels de valeurs.
            Format: {"nom_variable": {1: "Oui", 0: "Non"}}
            Les clés peuvent être int ou float (Stata accepte les deux).
    """
    _ensure_dir(path)
    # pyreadstat nécessite que les colonnes du DataFrame correspondent aux clés des labels
    pyreadstat.write_dta(
        df,
        path,
        variable_value_labels=labels or {},
    )


def export_latex(
    df: pd.DataFrame,
    path: str,
    caption: str = "",
    label: str = "",
    index: bool = False,
) -> None:
    """Exporte un DataFrame vers un fichier LaTeX (.tex).

    Args:
        df: Le DataFrame à exporter.
        path: Chemin de destination du fichier LaTeX.
        caption: Légende du tableau (optionnel).
        label: Label LaTeX pour la référence croisée (optionnel).
        index: Si True, écrit les index des lignes. Par défaut False.
    """
    _ensure_dir(path)
    latex_content = df.to_latex(
        index=index,
        caption=caption if caption else None,
        label=label if label else None,
        escape=True,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex_content)


def export_html(
    df: pd.DataFrame, path: str, title: str = "Tableau de données", index: bool = False
) -> None:
    """Exporte un DataFrame vers un fichier HTML.

    Args:
        df: Le DataFrame à exporter.
        path: Chemin de destination du fichier HTML.
        title: Titre de la page HTML. Par défaut "Tableau de données".
        index: Si True, écrit les index des lignes. Par défaut False.
    """
    _ensure_dir(path)
    html_table = df.to_html(index=index, escape=True, classes="dataframe")
    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: sans-serif; margin: 2rem; }}
        table.dataframe {{ border-collapse: collapse; width: 100%; }}
        table.dataframe th, table.dataframe td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        table.dataframe th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {html_table}
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)
