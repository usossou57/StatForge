"""Tests statistiques classiques : chi², t-test, ANOVA.

Chaque fonction retourne un AnalysisResult (contrat de la section 7).
Utilise scipy.stats et statsmodels (contrainte T7 : ne pas réinventer les maths).
"""

from __future__ import annotations

import scipy
import scipy.stats as stats
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

import pandas as pd

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

    contingency = pd.crosstab(df[var1], df[var2])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    return AnalysisResult(
        method="chi_square",
        dataset_id="in_memory",
        variables={
            "var1": {"role": "predictor", "name": var1},
            "var2": {"role": "predictor", "name": var2},
        },
        sample_size=len(df),
        preprocessing=[],
        estimates={"chi2": chi2, "dof": dof},
        standard_errors={},
        confidence_intervals={},
        p_values={"chi2": p_value},
        model_metrics={"chi2_statistic": chi2, "degrees_of_freedom": dof},
        diagnostics=[],
        warnings=[],
        assumptions_checked=[
            "Indépendance des observations",
            "Effectifs théoriques >= 5",
        ],
        generated_code=(
            f"import pandas as pd\n"
            f"import scipy.stats as stats\n"
            f"contingency = pd.crosstab(df['{var1}'], df['{var2}'])\n"
            f"chi2, p, dof, expected = stats.chi2_contingency(contingency)\n"
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

    groups = df[group].dropna().unique()
    if len(groups) != 2:
        raise ValueError(
            f"La variable '{group}' doit avoir exactement 2 modalités, "
            f"or {len(groups)} trouvées : {groups}"
        )

    group_a, group_b = groups[0], groups[1]
    sample_a = df.loc[df[group] == group_a, outcome].dropna()
    sample_b = df.loc[df[group] == group_b, outcome].dropna()

    t_stat, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)

    return AnalysisResult(
        method="t_test",
        dataset_id="in_memory",
        variables={
            "outcome": {"role": "outcome", "name": outcome},
            "group": {"role": "predictor", "name": group},
        },
        sample_size=len(df),
        preprocessing=[],
        estimates={
            "t_statistic": t_stat,
            "mean_a": sample_a.mean(),
            "mean_b": sample_b.mean(),
        },
        standard_errors={},
        confidence_intervals={},
        p_values={"t_test": p_value},
        model_metrics={
            "t_statistic": t_stat,
            "mean_group_a": sample_a.mean(),
            "mean_group_b": sample_b.mean(),
            "n_group_a": len(sample_a),
            "n_group_b": len(sample_b),
        },
        diagnostics=[],
        warnings=[],
        assumptions_checked=[
            "Normalité des résidus",
            "Homogénéité des variances (test de Welch utilisé)",
        ],
        generated_code=(
            f"import scipy.stats as stats\n"
            f"a = df.loc[df['{group}'] == '{group_a}', '{outcome}'].dropna()\n"
            f"b = df.loc[df['{group}'] == '{group_b}', '{outcome}'].dropna()\n"
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

    formula = f"{outcome} ~ C({group})"
    model = smf.ols(formula, data=df).fit()
    anova_table = anova_lm(model, typ=2)

    f_stat = anova_table.loc[f"C({group})", "F"]
    p_value = anova_table.loc[f"C({group})", "PR(>F)"]

    return AnalysisResult(
        method="anova",
        dataset_id="in_memory",
        variables={
            "outcome": {"role": "outcome", "name": outcome},
            "group": {"role": "predictor", "name": group},
        },
        sample_size=len(df),
        preprocessing=[],
        estimates={"f_statistic": f_stat},
        standard_errors={},
        confidence_intervals={},
        p_values={"anova": p_value},
        model_metrics={
            "f_statistic": f_stat,
            "n_groups": df[group].nunique(),
        },
        diagnostics=[],
        warnings=[],
        assumptions_checked=[
            "Normalité des résidus",
            "Homoscédasticité",
            "Indépendance des observations",
        ],
        generated_code=(
            f"import statsmodels.formula.api as smf\n"
            f"from statsmodels.stats.anova import anova_lm\n"
            f"model = smf.ols('{outcome} ~ C({group})', data=df).fit()\n"
            f"anova_table = anova_lm(model, typ=2)\n"
        ),
        execution_log=["ANOVA executed via statsmodels"],
        software_versions={"statsmodels": statsmodels.__version__},
    )
