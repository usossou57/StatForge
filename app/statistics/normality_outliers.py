"""Tests de normalité et détection d'outliers.

Ce module fournit des fonctions pour :
- Tester la normalité d'une distribution (Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling)
- Détecter les outliers (IQR, z-score, Mahalanobis multivarié)

Conforme à la spécification V2 §9.0 T16.3.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import-untyped]

from app.diagnostics.models import DiagnosticFinding, Severity
from app.results.models import AnalysisResult
from app.statistics.registry import MethodSpec, register_method

# ==============================================================================
# Tests de normalité
# ==============================================================================


def run_normality_tests(
    df: pd.DataFrame,
    column: str,
    methods: list[str] | None = None,
) -> AnalysisResult:
    """Exécute des tests de normalité sur une variable numérique.

    Args:
        df: DataFrame contenant les données.
        column: Nom de la colonne numérique à tester.
        methods: Liste des méthodes à exécuter. Par défaut : toutes.
            Options : "shapiro", "dagostino", "anderson".

    Returns:
        AnalysisResult contenant les statistiques de test, p-values et diagnostic.
        Diagnostic NON_NORMAL_DISTRIBUTION levé si au moins un test est significatif (p < 0.05).

    Raises:
        ValueError: Si la colonne n'existe pas ou n'est pas numérique.
    """
    if column not in df.columns:
        raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"La colonne '{column}' doit être numérique.")

    valid_methods = ("shapiro", "dagostino", "anderson")
    if methods is None:
        methods = list(valid_methods)
    else:
        invalid = [m for m in methods if m not in valid_methods]
        if invalid:
            raise ValueError(
                f"Méthodes invalides : {invalid}. Choisir parmi {valid_methods}."
            )

    series = df[column].dropna()
    n = len(series)
    values = series.to_numpy()

    test_results: dict[str, dict[str, float]] = {}
    is_normal = True
    warnings: list[str] = []

    # Shapiro-Wilk (valide pour n <= 5000)
    if "shapiro" in methods:
        if n < 3:
            warnings.append("Shapiro-Wilk : n < 3, test non réalisable.")
            test_results["shapiro"] = {
                "statistic": float("nan"),
                "p_value": float("nan"),
            }
        elif n > 5000:
            warnings.append(
                "Shapiro-Wilk : n > 5000, test non recommandé (utiliser dagostino ou anderson)."
            )
            stat, p = stats.shapiro(values[:5000])
            test_results["shapiro"] = {"statistic": float(stat), "p_value": float(p)}
            if p < 0.05:
                is_normal = False
        else:
            stat, p = stats.shapiro(values)
            test_results["shapiro"] = {"statistic": float(stat), "p_value": float(p)}
            if p < 0.05:
                is_normal = False

    # D'Agostino-Pearson (nécessite n >= 20)
    if "dagostino" in methods:
        if n < 20:
            warnings.append("D'Agostino-Pearson : n < 20, test non réalisable.")
            test_results["dagostino"] = {
                "statistic": float("nan"),
                "p_value": float("nan"),
            }
        else:
            stat, p = stats.normaltest(values)
            test_results["dagostino"] = {"statistic": float(stat), "p_value": float(p)}
            if p < 0.05:
                is_normal = False

    # Anderson-Darling
    if "anderson" in methods:
        try:
            # SciPy >= 1.17 : la nouvelle API retourne directement la p-value
            result = stats.anderson(values, dist="norm", method="interpolate")
            stat_ad = float(result.statistic)
            p_approx = float(result.pvalue)
            critical_5pct = float("nan")
        except TypeError:
            # SciPy < 1.17 : ancienne API avec critical_values
            result = stats.anderson(values, dist="norm")
            critical_5pct = float(result.critical_values[2])
            stat_ad = float(result.statistic)
            p_approx = _anderson_p_value(
                stat_ad, result.critical_values, result.significance_level
            )

        test_results["anderson"] = {
            "statistic": stat_ad,
            "p_value": p_approx,
            "critical_value_5pct": critical_5pct,
        }
        # Si critical_5pct est NaN (nouvelle API), on se fie à la p-value
        if (
            not np.isnan(critical_5pct)
            and stat_ad > critical_5pct
            or np.isnan(critical_5pct)
            and p_approx < 0.05
        ):
            is_normal = False

    # Statistiques descriptives pour contexte
    descriptive = {
        "n": int(n),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "skewness": float(series.skew()),  # type: ignore[arg-type]
        "kurtosis": float(series.kurtosis()),  # type: ignore[arg-type]
    }

    # Diagnostic
    diagnostics: list[DiagnosticFinding] = []
    if not is_normal:
        diagnostics.append(
            DiagnosticFinding(
                code="NON_NORMAL_DISTRIBUTION",
                severity=Severity.WARNING,
                message=(
                    f"La variable '{column}' ne suit pas une distribution normale "
                    f"(au moins un test significatif à alpha=0.05)."
                ),
            )
        )

    return AnalysisResult(
        method="run_normality_tests",
        dataset_id="in_memory",
        variables={"column": column, "methods": methods},
        sample_size=n,
        preprocessing=["Suppression des valeurs manquantes"],
        estimates={"descriptive": descriptive},
        standard_errors={},
        confidence_intervals={},
        p_values={name: res["p_value"] for name, res in test_results.items()},
        model_metrics={
            "method_family": "diagnostic",
            "tables": {"normality_tests": test_results},
            "is_normal": is_normal,
        },
        diagnostics=diagnostics,
        warnings=warnings,
        assumptions_checked=["Variable numérique continue"],
        generated_code=(
            f"from scipy import stats\n"
            f"import pandas as pd\n"
            f"# df = pd.read_csv('votre_fichier.csv')\n"
            f"values = df['{column}'].dropna().values\n"
            f"print(stats.shapiro(values))\n"
            f"print(stats.normaltest(values))\n"
            f"print(stats.anderson(values, dist='norm'))"
        ),
        execution_log=[f"Tests de normalité exécutés sur '{column}' (n={n})"],
        software_versions={
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    )


def _anderson_p_value(
    stat: float, critical_values: np.ndarray, significance_levels: np.ndarray
) -> float:
    """Calcule une p-value approximative pour le test d'Anderson-Darling (SciPy < 1.17)."""
    if stat <= critical_values[0]:
        return float(significance_levels[0]) / 100.0
    if stat >= critical_values[-1]:
        return float(significance_levels[-1]) / 100.0
    for i in range(len(critical_values) - 1):
        if critical_values[i] <= stat <= critical_values[i + 1]:
            frac = (stat - critical_values[i]) / (
                critical_values[i + 1] - critical_values[i]
            )
            p = significance_levels[i] - frac * (
                significance_levels[i] - significance_levels[i + 1]
            )
            return float(p) / 100.0
    return 0.05


# ==============================================================================
# Détection d'outliers
# ==============================================================================


def detect_outliers(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "iqr",
    threshold: float | None = None,
) -> AnalysisResult:
    """Détecte les outliers dans une ou plusieurs variables."""
    valid_methods = ("iqr", "zscore", "mahalanobis")
    if method not in valid_methods:
        raise ValueError(
            f"Méthode '{method}' non reconnue. Choisir parmi {valid_methods}."
        )

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Colonnes non numériques : {non_numeric}")

    subset = df[columns].dropna()
    n = len(subset)

    outlier_indices: list[int] = []
    # Typage explicite en Any pour supporter la structure différente de Mahalanobis
    outlier_details: dict[str, Any] = {}
    used_threshold = threshold

    if method == "iqr":
        if used_threshold is None:
            used_threshold = 1.5
        for col in columns:
            series = subset[col]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - used_threshold * iqr
            upper = q3 + used_threshold * iqr
            mask = (series < lower) | (series > upper)
            col_outliers = series[mask].index.tolist()
            outlier_indices.extend(col_outliers)
            outlier_details[col] = {
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "n_outliers": len(col_outliers),
                "outlier_indices": col_outliers,
            }

    elif method == "zscore":
        if used_threshold is None:
            used_threshold = 3.0
        for col in columns:
            series = subset[col]
            mean = series.mean()
            std = series.std()
            if std == 0:
                outlier_details[col] = {
                    "mean": float(mean),
                    "std": 0.0,
                    "n_outliers": 0,
                    "outlier_indices": [],
                }
                continue
            zscores = (series - mean).abs() / std
            mask = zscores > used_threshold
            col_outliers = series[mask].index.tolist()
            outlier_indices.extend(col_outliers)
            outlier_details[col] = {
                "mean": float(mean),
                "std": float(std),
                "threshold": float(used_threshold),
                "n_outliers": len(col_outliers),
                "outlier_indices": col_outliers,
            }

    else:  # mahalanobis
        if len(columns) < 2:
            raise ValueError(
                "La méthode de Mahalanobis nécessite au moins 2 variables."
            )
        if used_threshold is None:
            used_threshold = float(stats.chi2.ppf(0.975, df=len(columns)))
        data = subset[columns].to_numpy()
        mean = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)
        diff = data - mean
        mahal_sq = np.sum(diff @ cov_inv * diff, axis=1)
        mask = mahal_sq > used_threshold
        outlier_indices = subset.index[mask].tolist()
        outlier_details = {
            "method": "mahalanobis",
            "threshold": float(used_threshold),
            "n_outliers": int(mask.sum()),
            "outlier_indices": outlier_indices,
            "max_distance_squared": float(mahal_sq.max()) if len(mahal_sq) > 0 else 0.0,
        }

    outlier_indices_unique = sorted(set(outlier_indices))
    n_outliers = len(outlier_indices_unique)

    diagnostics: list[DiagnosticFinding] = []
    if n_outliers > 0:
        diagnostics.append(
            DiagnosticFinding(
                code="OUTLIERS_DETECTED",
                severity=Severity.INFORMATION,
                message=(
                    f"{n_outliers} outlier(s) détecté(s) "
                    f"({n_outliers / n * 100:.1f}% des observations) "
                    f"via la méthode '{method}'."
                ),
            )
        )

    return AnalysisResult(
        method="detect_outliers",
        dataset_id="in_memory",
        variables={"columns": columns, "method": method, "threshold": used_threshold},
        sample_size=n,
        preprocessing=["Suppression des lignes avec valeurs manquantes"],
        estimates={
            "n_outliers": n_outliers,
            "outlier_indices": outlier_indices_unique,
        },
        standard_errors={},
        confidence_intervals={},
        p_values={},
        model_metrics={
            "method_family": "diagnostic",
            "tables": {"outlier_details": outlier_details},
        },
        diagnostics=diagnostics,
        warnings=[],
        assumptions_checked=[
            "Variables numériques",
            "Distribution approximativement normale (z-score, Mahalanobis)",
        ],
        generated_code=(
            f"import numpy as np\n"
            f"import pandas as pd\n"
            f"from scipy import stats\n"
            f"# df = pd.read_csv('votre_fichier.csv')\n"
            f"# Méthode : {method}, Seuil : {used_threshold}\n"
        ),
        execution_log=[
            (
                f"Détection d'outliers via '{method}' sur {len(columns)} variable(s), "
                f"{n_outliers} outlier(s) trouvé(s)."
            )
        ],
        software_versions={
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    )


# ==============================================================================
# Enregistrement dans le registre (V2 §7.1)
# ==============================================================================
register_method(
    MethodSpec(
        method_id="run_normality_tests",
        label_fr="Tests de normalité",
        family="diagnostic",
        engine="python",
        outcome_types=["numeric"],
        predictor_types=[],
        min_sample_size=3,
        assumptions=["Variable numérique continue"],
        reference_impl="scipy.stats.shapiro / normaltest / anderson",
        r_reference="shapiro.test / nortest::ad.test",
        phase=16,
    ),
    run_normality_tests,
)

register_method(
    MethodSpec(
        method_id="detect_outliers",
        label_fr="Détection d'outliers",
        family="diagnostic",
        engine="python",
        outcome_types=["numeric"],
        predictor_types=[],
        min_sample_size=4,
        assumptions=["Variables numériques", "Normalité (z-score, Mahalanobis)"],
        reference_impl="numpy / scipy.stats",
        r_reference="outliers::outlier / MVoutlier::pcout",
        phase=16,
    ),
    detect_outliers,
)
