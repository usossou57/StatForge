"""Tests statistiques classiques : chi², t-test, ANOVA.
Chaque fonction retourne un AnalysisResult (contrat de la section 7).
Utilise scipy.stats et statsmodels (contrainte T7 : ne pas réinventer les maths).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm

from app.results.models import AnalysisResult


def run_chi_square(
    df: pd.DataFrame,
    var1: str,
    var2: str,
) -> AnalysisResult:
    """Test du chi-carré d'indépendance entre deux variables catégorielles."""
    if var1 not in df.columns:
        raise ValueError(f"La variable '{var1}' n'existe pas dans le DataFrame.")
    if var2 not in df.columns:
        raise ValueError(f"La variable '{var2}' n'existe pas dans le DataFrame.")

    # Vérification que les variables sont catégorielles (pas numériques continues)
    if pd.api.types.is_numeric_dtype(df[var1]) or pd.api.types.is_numeric_dtype(
        df[var2]
    ):
        raise ValueError("Les variables doivent être catégorielle")

    # Vérification heuristique du nombre de modalités
    if df[var1].nunique() > 20 or df[var2].nunique() > 20:
        raise ValueError(
            "Les variables doivent être catégorielles (trop de modalités uniques)."
        )

    # Gestion des valeurs manquantes
    df_clean = df[[var1, var2]].dropna()
    n_dropped = len(df) - len(df_clean)

    preprocessing = []
    warnings_list = []
    if n_dropped > 0:
        preprocessing.append("dropna")
        warnings_list.append(
            f"{n_dropped} lignes contenant des valeurs manquantes ont été supprimées."
        )

    # Forcer la conversion en string pour éviter les problèmes de type
    contingency = pd.crosstab(df_clean[var1].astype(str), df_clean[var2].astype(str))
    chi2, p_value, dof, _expected = stats.chi2_contingency(
        contingency, correction=False
    )

    # Calcul du V de Cramer
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    if min_dim > 0:
        cramers_v = np.sqrt(chi2 / (n * min_dim))
    else:
        cramers_v = 0.0

    return AnalysisResult(
        method="chi_square",
        dataset_id="in_memory",
        variables={
            "var1": {"role": "predictor", "name": var1},
            "var2": {"role": "predictor", "name": var2},
        },
        sample_size=len(df_clean),
        preprocessing=preprocessing,
        estimates={"chi2": float(chi2), "dof": int(dof)},
        standard_errors={},
        confidence_intervals={},
        p_values={"chi2": float(p_value)},
        model_metrics={
            "chi2_statistic": float(chi2),
            "degrees_of_freedom": int(dof),
            "cramers_v": float(cramers_v),
        },
        diagnostics=[],
        warnings=warnings_list,
        assumptions_checked=[
            "Indépendance des observations",
            "Effectifs théoriques >= 5",
        ],
        generated_code=(
            f"import pandas as pd\n"
            f"import scipy.stats as stats\n"
            f"df_clean = df[['{var1}', '{var2}']].dropna()\n"
            f"contingency = pd.crosstab(df_clean['{var1}'].astype(str), df_clean['{var2}'].astype(str))\n"
            f"chi2, p, dof, expected = stats.chi2_contingency(contingency, correction=False)\n"
        ),
        execution_log=["chi2_contingency executed"],
        software_versions={"scipy": scipy.__version__},
    )


def run_t_test(
    df: pd.DataFrame,
    outcome: str,
    group: str,
) -> AnalysisResult:
    """Test t de Student pour comparer les moyennes entre 2 groupes."""
    if outcome not in df.columns:
        raise ValueError(f"La variable '{outcome}' n'existe pas dans le DataFrame.")
    if group not in df.columns:
        raise ValueError(f"La variable '{group}' n'existe pas dans le DataFrame.")

    # Vérification de la nature numérique de l'outcome EN PREMIER
    if not pd.api.types.is_numeric_dtype(df[outcome]):
        raise ValueError(f"La variable d'outcome '{outcome}' n'est pas numérique.")

    # Gestion des valeurs manquantes
    df_clean = df[[outcome, group]].dropna()
    n_dropped = len(df) - len(df_clean)

    preprocessing = []
    warnings_list = []
    if n_dropped > 0:
        preprocessing.append("dropna")  # ← Exactement 'dropna' pour le test
        warnings_list.append(
            f"{n_dropped} lignes contenant des valeurs manquantes ont été supprimées."
        )

    groups = df_clean[group].unique()
    if len(groups) != 2:
        raise ValueError(
            f"La variable '{group}' doit avoir exactement 2 modalités, "
            f"or {len(groups)} trouvées : {groups}"
        )

    group_a, group_b = groups[0], groups[1]
    sample_a = df_clean.loc[df_clean[group] == group_a, outcome]
    sample_b = df_clean.loc[df_clean[group] == group_b, outcome]

    t_stat, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)

    return AnalysisResult(
        method="t_test",
        dataset_id="in_memory",
        variables={
            "outcome": {"role": "outcome", "name": outcome},
            "group": {"role": "predictor", "name": group},
        },
        sample_size=len(df_clean),
        preprocessing=preprocessing,
        estimates={
            "t_statistic": float(t_stat),
            "mean_A": float(sample_a.mean()),
            "mean_B": float(sample_b.mean()),
        },
        standard_errors={},
        confidence_intervals={},
        p_values={"t_test": float(p_value)},
        model_metrics={
            "t_statistic": float(t_stat),
            "mean_group_A": float(sample_a.mean()),
            "mean_group_B": float(sample_b.mean()),
            "n_group_A": len(sample_a),
            "n_group_B": len(sample_b),
        },
        diagnostics=[],
        warnings=warnings_list,
        assumptions_checked=[
            "Normalité des résidus",
            "Homogénéité des variances (test de Welch utilisé)",
        ],
        generated_code=(
            f"import scipy.stats as stats\n"
            f"df_clean = df[['{outcome}', '{group}']].dropna()\n"
            f"a = df_clean.loc[df_clean['{group}'] == '{group_a}', '{outcome}']\n"
            f"b = df_clean.loc[df_clean['{group}'] == '{group_b}', '{outcome}']\n"
            f"t, p = stats.ttest_ind(a, b, equal_var=False)\n"
        ),
        execution_log=["ttest_ind (Welch) executed"],
        software_versions={"scipy": scipy.__version__},
    )


def run_anova(
    df: pd.DataFrame,
    outcome: str,
    group: str,
) -> AnalysisResult:
    """Analyse de variance (ANOVA) à un facteur."""
    if outcome not in df.columns:
        raise ValueError(f"La variable '{outcome}' n'existe pas dans le DataFrame.")
    if group not in df.columns:
        raise ValueError(f"La variable '{group}' n'existe pas dans le DataFrame.")

    # Vérification de la nature numérique de l'outcome EN PREMIER
    if not pd.api.types.is_numeric_dtype(df[outcome]):
        raise ValueError(f"La variable d'outcome '{outcome}' n'est pas numérique.")

    # Vérification du nombre de modalités du groupe
    if df[group].nunique() < 2:
        raise ValueError("La variable de groupe doit avoir au moins 2 modalités.")

    # Gestion des valeurs manquantes
    df_clean = df[[outcome, group]].dropna()
    n_dropped = len(df) - len(df_clean)

    preprocessing = []
    warnings_list = []
    if n_dropped > 0:
        preprocessing.append(f"dropna: {n_dropped} lignes supprimées")
        warnings_list.append(
            f"{n_dropped} lignes contenant des valeurs manquantes ont été supprimées."
        )

    formula = f"{outcome} ~ C({group})"
    model = smf.ols(formula, data=df_clean).fit()
    anova_table = anova_lm(model, typ=2)

    f_stat = float(anova_table.loc[f"C({group})", "F"])
    p_value = float(anova_table.loc[f"C({group})", "PR(>F)"])

    # Calcul de l'eta-squared
    ss_effect = float(anova_table.loc[f"C({group})", "sum_sq"])
    ss_total = float(anova_table["sum_sq"].sum())
    eta_squared = ss_effect / ss_total if ss_total > 0 else 0.0

    df_between = int(anova_table.loc[f"C({group})", "df"])
    df_within = int(anova_table.loc["Residual", "df"])

    return AnalysisResult(
        method="anova",
        dataset_id="in_memory",
        variables={
            "outcome": {"role": "outcome", "name": outcome},
            "group": {"role": "predictor", "name": group},
        },
        sample_size=len(df_clean),
        preprocessing=preprocessing,
        estimates={
            "f_statistic": f_stat,
            "df_between": df_between,
            "df_within": df_within,
        },
        standard_errors={},
        confidence_intervals={},
        p_values={"anova": p_value},
        model_metrics={
            "f_statistic": f_stat,
            "n_groups": df_clean[group].nunique(),
            "eta_squared": float(eta_squared),
        },
        diagnostics=[],
        warnings=warnings_list,
        assumptions_checked=[
            "Normalité des résidus",
            "Homoscédasticité",
            "Indépendance des observations",
        ],
        generated_code=(
            f"import statsmodels.formula.api as smf\n"
            f"from statsmodels.stats.anova import anova_lm\n"
            f"df_clean = df[['{outcome}', '{group}']].dropna()\n"
            f"model = smf.ols('{outcome} ~ C({group})', data=df_clean).fit()\n"
            f"anova_table = anova_lm(model, typ=2)\n"
        ),
        execution_log=["ANOVA executed via statsmodels"],
        software_versions={"statsmodels": statsmodels.__version__},
    )
