"""Statistiques descriptives et tableaux croisés.
Fonctions utilitaires pour l'exploration des données avant analyse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy  # type: ignore[import-untyped]

from app.results.models import AnalysisResult
from app.statistics.registry import MethodSpec, register_method


def create_contingency_table(df: pd.DataFrame, row: str, col: str) -> pd.DataFrame:
    """Crée un tableau croisé (contingence) entre deux variables catégorielles."""
    if row not in df.columns:
        raise ValueError(f"La variable '{row}' n'existe pas dans le DataFrame.")
    if col not in df.columns:
        raise ValueError(f"La variable '{col}' n'existe pas dans le DataFrame.")

    return pd.crosstab(df[row], df[col])


def _safe_round(val: float | None, decimals: int = 4) -> float | None:
    """Arrondit une valeur en gérant proprement les NaN et None."""
    if val is None or pd.isna(val):
        return None
    return float(round(float(val), decimals))


def frequency_table(
    df: pd.DataFrame, column: str, weights: str | None = None
) -> AnalysisResult:
    """Calcule les fréquences d'une variable, avec ou sans pondération."""
    if column not in df.columns:
        raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame.")

    if weights and weights not in df.columns:
        raise ValueError(
            f"La colonne de poids '{weights}' n'existe pas dans le DataFrame."
        )

    if weights:
        freq_series = df.groupby(column)[weights].sum()
        total_weight = freq_series.sum()
        pct_series = (
            (freq_series / total_weight * 100).round(2)
            if total_weight > 0
            else freq_series * 0
        )
    else:
        freq_series = df[column].value_counts(dropna=False)
        total_weight = len(df)
        pct_series = (
            (freq_series / total_weight * 100).round(2)
            if total_weight > 0
            else freq_series * 0
        )

    table_data = []
    for val, count in freq_series.items():
        table_data.append(
            {
                "value": str(val),
                "count": float(count) if weights else int(count),
                "percentage": float(pct_series.get(val, 0.0)),
            }
        )

    return AnalysisResult(
        method="frequency_table",
        dataset_id="in_memory",
        variables={"target": column, "weights": weights},
        sample_size=len(df),
        preprocessing=["Calcul pondéré" if weights else "Calcul non pondéré"],
        estimates={"total_observations": len(df)},
        standard_errors={},
        confidence_intervals={},
        p_values={},
        model_metrics={
            "method_family": "descriptive",
            "tables": {"frequency": table_data},
        },
        diagnostics=[],
        warnings=[],
        assumptions_checked=[],
        generated_code=f"# Code reproductible pour frequency_table\nimport pandas as pd\n# df = pd.read_csv('votre_fichier.csv')\nprint(df['{column}'].value_counts(dropna=False))",
        execution_log=["Calcul des fréquences effectué"],
        software_versions={"pandas": pd.__version__, "numpy": np.__version__},
    )


def summarize_variables(
    df: pd.DataFrame, columns: list[str] | None = None
) -> AnalysisResult:
    """Calcule des résumés descriptifs étendus pour une ou plusieurs variables."""
    if columns is None:
        columns = df.select_dtypes(include=["number", "bool"]).columns.tolist()

    summary_data = []
    for col in columns:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        n = len(series)
        n_missing = len(df[col]) - n
        total_len = len(df[col])
        missing_pct = (n_missing / total_len * 100) if total_len > 0 else 0.0

        if pd.api.types.is_numeric_dtype(df[col]):
            # Conversions explicites avec ignore pour satisfaire mypy sur les scalaires pandas
            min_val = float(series.min()) if pd.notna(series.min()) else None  # type: ignore[arg-type]
            max_val = float(series.max()) if pd.notna(series.max()) else None  # type: ignore[arg-type]

            mode_val = series.mode().iloc[0] if not series.mode().empty else None
            mode_float = (
                float(mode_val) if mode_val is not None and pd.notna(mode_val) else None
            )  # type: ignore[arg-type]

            summary_data.append(
                {
                    "variable": col,
                    "n": int(n),
                    "n_missing": int(n_missing),
                    "missing_pct": _safe_round(float(missing_pct)),
                    "mean": _safe_round(
                        float(series.mean()) if pd.notna(series.mean()) else None
                    ),  # type: ignore[arg-type]
                    "median": _safe_round(
                        float(series.median()) if pd.notna(series.median()) else None
                    ),  # type: ignore[arg-type]
                    "mode": _safe_round(mode_float),
                    "std": _safe_round(
                        float(series.std()) if pd.notna(series.std()) else None
                    ),  # type: ignore[arg-type]
                    "variance": _safe_round(
                        float(series.var()) if pd.notna(series.var()) else None
                    ),  # type: ignore[arg-type]
                    "min": _safe_round(min_val),
                    "max": _safe_round(max_val),
                    "range": _safe_round(max_val - min_val)
                    if n > 0 and min_val is not None and max_val is not None
                    else None,
                    "q1": _safe_round(
                        float(series.quantile(0.25))
                        if pd.notna(series.quantile(0.25))
                        else None
                    ),  # type: ignore[arg-type]
                    "q3": _safe_round(
                        float(series.quantile(0.75))
                        if pd.notna(series.quantile(0.75))
                        else None
                    ),  # type: ignore[arg-type]
                    "iqr": _safe_round(
                        float(series.quantile(0.75) - series.quantile(0.25))
                    )
                    if n > 0
                    else None,  # type: ignore[arg-type]
                    "skewness": _safe_round(
                        float(series.skew()) if pd.notna(series.skew()) else None
                    ),  # type: ignore[arg-type]
                    "kurtosis": _safe_round(
                        float(series.kurtosis())
                        if pd.notna(series.kurtosis())
                        else None
                    ),  # type: ignore[arg-type]
                }
            )
        else:
            mode_val = series.mode().iloc[0] if not series.mode().empty else None
            summary_data.append(
                {
                    "variable": col,
                    "n": int(n),
                    "n_missing": int(n_missing),
                    "missing_pct": _safe_round(float(missing_pct)),
                    "n_unique": int(series.nunique()),
                    "mode": str(mode_val) if mode_val is not None else None,
                }
            )

    return AnalysisResult(
        method="summarize_variables",
        dataset_id="in_memory",
        variables={"columns": columns},
        sample_size=len(df),
        preprocessing=[f"Analyse de {len(columns)} colonnes"],
        estimates={},
        standard_errors={},
        confidence_intervals={},
        p_values={},
        model_metrics={
            "method_family": "descriptive",
            "tables": {"summary": summary_data},
        },
        diagnostics=[],
        warnings=[],
        assumptions_checked=[],
        generated_code="# Code reproductible pour summarize_variables\nimport pandas as pd\n# df = pd.read_csv('votre_fichier.csv')\nprint(df.describe(include='all'))",
        execution_log=["Résumé descriptif calculé"],
        software_versions={
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    )


# ==============================================================================
# Enregistrement dans le registre (V2 §7.1)
# ==============================================================================
register_method(
    MethodSpec(
        method_id="frequency_table",
        label_fr="Tableau de fréquences",
        family="descriptive",
        engine="python",
        outcome_types=["categorical", "numeric", "binary", "ordinal"],
        predictor_types=[],
        min_sample_size=1,
        assumptions=["Aucune hypothèse spécifique"],
        reference_impl="pandas.value_counts",
        r_reference="table",
        phase=16,
    ),
    frequency_table,
)

register_method(
    MethodSpec(
        method_id="summarize_variables",
        label_fr="Résumé descriptif étendu",
        family="descriptive",
        engine="python",
        outcome_types=["numeric", "categorical", "binary", "ordinal", "datetime"],
        predictor_types=[],
        min_sample_size=1,
        assumptions=["Aucune hypothèse spécifique"],
        reference_impl="pandas.describe",
        r_reference="summary",
        phase=16,
    ),
    summarize_variables,
)
