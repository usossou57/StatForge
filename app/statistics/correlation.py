"""Calculs de corrélations, covariances et corrélations partielles.

Fonctions pour l'analyse des relations linéaires et monotones entre variables,
avec p-values et intervalles de confiance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import-untyped]

from app.results.models import AnalysisResult
from app.statistics.registry import MethodSpec, register_method


def run_correlation(
    df: pd.DataFrame,
    variables: list[str],
    method: str = "pearson",
) -> AnalysisResult:
    """Calcule une matrice de corrélation avec p-values et IC (Fisher pour Pearson).

    Args:
        df: DataFrame contenant les données.
        variables: Liste des colonnes numériques à corréler.
        method: Méthode de corrélation ("pearson", "spearman", "kendall").

    Returns:
        AnalysisResult avec matrices de corrélation, p-values et IC.

    Raises:
        ValueError: Si la méthode n'est pas reconnue ou si des colonnes manquent.
    """
    valid_methods = ("pearson", "spearman", "kendall")
    if method not in valid_methods:
        raise ValueError(
            f"Méthode '{method}' non reconnue. Choisir parmi {valid_methods}."
        )

    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le DataFrame : {missing}")

    subset = df[variables].dropna()
    n = len(subset)
    k = len(variables)

    # Matrice de corrélation
    corr_matrix = subset.corr(method=method)  # type: ignore[arg-type]

    # Calcul des p-values et IC par paire
    pval_matrix = pd.DataFrame(np.ones((k, k)), index=variables, columns=variables)
    ci_lower_matrix = pd.DataFrame(
        np.full((k, k), np.nan), index=variables, columns=variables
    )
    ci_upper_matrix = pd.DataFrame(
        np.full((k, k), np.nan), index=variables, columns=variables
    )

    for i, v1 in enumerate(variables):
        for j, v2 in enumerate(variables):
            if i >= j:
                continue
            x = subset[v1].values
            y = subset[v2].values

            if method == "pearson":
                r, p = stats.pearsonr(x, y)
                # IC via transformation de Fisher
                z = np.arctanh(r)
                se = 1.0 / np.sqrt(n - 3) if n > 3 else float("inf")
                z_crit = stats.norm.ppf(0.975)
                ci_lower_matrix.loc[v1, v2] = np.tanh(z - z_crit * se)
                ci_upper_matrix.loc[v1, v2] = np.tanh(z + z_crit * se)
                ci_lower_matrix.loc[v2, v1] = ci_lower_matrix.loc[v1, v2]
                ci_upper_matrix.loc[v2, v1] = ci_upper_matrix.loc[v1, v2]
            elif method == "spearman":
                r, p = stats.spearmanr(x, y)
            else:  # kendall
                r, p = stats.kendalltau(x, y)

            pval_matrix.loc[v1, v2] = p
            pval_matrix.loc[v2, v1] = p

    # Diagonale : p-value = 0 (boucle explicite pour éviter les problèmes de type)
    for v in variables:
        pval_matrix.loc[v, v] = 0.0

    return AnalysisResult(
        method="run_correlation",
        dataset_id="in_memory",
        variables={"columns": variables, "method": method},
        sample_size=n,
        preprocessing=[f"Corrélation {method} sur {k} variables"],
        estimates={
            "correlation_matrix": corr_matrix.to_dict(),
        },
        standard_errors={},
        confidence_intervals={
            "ci_lower": ci_lower_matrix.to_dict(),
            "ci_upper": ci_upper_matrix.to_dict(),
        },
        p_values={"pvalue_matrix": pval_matrix.to_dict()},
        model_metrics={
            "method_family": "descriptive",
            "tables": {
                "correlation": corr_matrix.to_dict(),
                "pvalue": pval_matrix.to_dict(),
                "ci_lower": ci_lower_matrix.to_dict(),
                "ci_upper": ci_upper_matrix.to_dict(),
            },
        },
        diagnostics=[],
        warnings=[] if n > 3 else ["n <= 3 : IC de Fisher non fiable"],
        assumptions_checked=[],
        generated_code=(
            f"# Code reproductible pour run_correlation\n"
            f"import pandas as pd\n"
            f"# df = pd.read_csv('votre_fichier.csv')\n"
            f"print(df[{variables}].corr(method='{method}'))"
        ),
        execution_log=[f"Corrélation {method} calculée sur {n} observations"],
        software_versions={
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    )


def run_covariance(
    df: pd.DataFrame,
    variables: list[str],
) -> AnalysisResult:
    """Calcule la matrice de covariance entre variables numériques.

    Args:
        df: DataFrame contenant les données.
        variables: Liste des colonnes numériques.

    Returns:
        AnalysisResult avec la matrice de covariance.

    Raises:
        ValueError: Si des colonnes manquent.
    """
    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le DataFrame : {missing}")

    subset = df[variables].dropna()
    n = len(subset)
    cov_matrix = subset.cov()

    return AnalysisResult(
        method="run_covariance",
        dataset_id="in_memory",
        variables={"columns": variables},
        sample_size=n,
        preprocessing=[f"Covariance sur {len(variables)} variables"],
        estimates={"covariance_matrix": cov_matrix.to_dict()},
        standard_errors={},
        confidence_intervals={},
        p_values={},
        model_metrics={
            "method_family": "descriptive",
            "tables": {"covariance": cov_matrix.to_dict()},
        },
        diagnostics=[],
        warnings=[],
        assumptions_checked=[],
        generated_code=(
            f"# Code reproductible pour run_covariance\n"
            f"import pandas as pd\n"
            f"# df = pd.read_csv('votre_fichier.csv')\n"
            f"print(df[{variables}].cov())"
        ),
        execution_log=[f"Matrice de covariance calculée sur {n} observations"],
        software_versions={
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    )


def run_partial_correlation(
    df: pd.DataFrame,
    x: str,
    y: str,
    controls: list[str],
) -> AnalysisResult:
    """Calcule la corrélation partielle entre x et y en contrôlant pour d'autres variables.

    Utilise la méthode des résidus de régression linéaire.

    Args:
        df: DataFrame contenant les données.
        x: Nom de la première variable.
        y: Nom de la seconde variable.
        controls: Liste des variables de contrôle.

    Returns:
        AnalysisResult avec la corrélation partielle, p-value et IC.

    Raises:
        ValueError: Si des colonnes manquent.
    """
    all_cols = [x, y] + controls
    missing = [v for v in all_cols if v not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le DataFrame : {missing}")

    subset = df[all_cols].dropna()
    n = len(subset)

    # Régresser x et y sur les contrôles pour obtenir les résidus
    if controls:
        X_ctrl = subset[controls].values
        X_ctrl_with_intercept = np.column_stack([np.ones(n), X_ctrl])
        beta_x = np.linalg.lstsq(
            X_ctrl_with_intercept, subset[x].to_numpy(), rcond=None
        )[0]
        beta_y = np.linalg.lstsq(
            X_ctrl_with_intercept, subset[y].to_numpy(), rcond=None
        )[0]
        res_x = subset[x].to_numpy() - X_ctrl_with_intercept @ beta_x
        res_y = subset[y].to_numpy() - X_ctrl_with_intercept @ beta_y
    else:
        res_x = subset[x].to_numpy()
        res_y = subset[y].to_numpy()

    r_partial, p_value = stats.pearsonr(res_x, res_y)

    # IC via transformation de Fisher
    df_residual = n - len(controls) - 2
    if df_residual > 1:
        z = np.arctanh(r_partial)
        se = 1.0 / np.sqrt(df_residual)
        z_crit = stats.norm.ppf(0.975)
        ci_lower = float(np.tanh(z - z_crit * se))
        ci_upper = float(np.tanh(z + z_crit * se))
    else:
        ci_lower = float("nan")
        ci_upper = float("nan")

    return AnalysisResult(
        method="run_partial_correlation",
        dataset_id="in_memory",
        variables={"x": x, "y": y, "controls": controls},
        sample_size=n,
        preprocessing=[f"Corrélation partielle {x}~{y} contrôlée par {controls}"],
        estimates={"r_partial": float(r_partial)},
        standard_errors={},
        confidence_intervals={
            "r_partial": [ci_lower, ci_upper],
        },
        p_values={"r_partial": float(p_value)},
        model_metrics={
            "method_family": "descriptive",
            "tables": {
                "partial_correlation": {
                    "x": x,
                    "y": y,
                    "controls": controls,
                    "r": float(r_partial),
                    "p_value": float(p_value),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "df_residual": df_residual,
                }
            },
        },
        diagnostics=[],
        warnings=(["df résiduel <= 1 : IC non calculable"] if df_residual <= 1 else []),
        assumptions_checked=[],
        generated_code=(
            f"# Code reproductible pour run_partial_correlation\n"
            f"# Utiliser la méthode des résidus de régression\n"
            f"import numpy as np\nfrom scipy import stats\n"
            f"# res_x = residuals(lm({x} ~ {controls}))\n"
            f"# res_y = residuals(lm({y} ~ {controls}))\n"
            f"# stats.pearsonr(res_x, res_y)"
        ),
        execution_log=[
            f"Corrélation partielle calculée : r={r_partial:.4f}, p={p_value:.4e}"
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
        method_id="run_correlation",
        label_fr="Matrice de corrélations",
        family="descriptive",
        engine="python",
        outcome_types=["numeric"],
        predictor_types=["numeric"],
        min_sample_size=4,
        assumptions=["Normalité bivariée (Pearson)", "Monotonie (Spearman/Kendall)"],
        reference_impl="scipy.stats.pearsonr",
        r_reference="cor / cor.test",
        phase=16,
    ),
    run_correlation,
)

register_method(
    MethodSpec(
        method_id="run_covariance",
        label_fr="Matrice de covariances",
        family="descriptive",
        engine="python",
        outcome_types=["numeric"],
        predictor_types=["numeric"],
        min_sample_size=2,
        assumptions=["Aucune hypothèse spécifique"],
        reference_impl="pandas.DataFrame.cov",
        r_reference="cov",
        phase=16,
    ),
    run_covariance,
)

register_method(
    MethodSpec(
        method_id="run_partial_correlation",
        label_fr="Corrélation partielle",
        family="descriptive",
        engine="python",
        outcome_types=["numeric"],
        predictor_types=["numeric"],
        min_sample_size=5,
        assumptions=["Linéarité", "Normalité des résidus"],
        reference_impl="scipy.stats.pearsonr sur résidus",
        r_reference="ppcor::pcor",
        phase=16,
    ),
    run_partial_correlation,
)
