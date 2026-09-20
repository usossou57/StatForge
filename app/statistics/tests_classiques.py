"""Tests statistiques classiques : chi-deux, test t, ANOVA."""

import sys

import numpy as np
import pandas as pd
from scipy import stats

from app.results.models import AnalysisResult


def _get_software_versions() -> dict:
    """Retourne les versions des logiciels utilisés."""
    import pandas
    import scipy
    return {
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "python": sys.version.split()[0],
    }


def _is_categorical(series: pd.Series) -> bool:
    """Vérifie si une série pandas doit être traitée comme catégorielle."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        return True
    return bool(pd.api.types.is_numeric_dtype(series) and series.nunique() <= 5)


def run_chi_square(df: pd.DataFrame, var1: str, var2: str, dataset_id: str = "unknown") -> AnalysisResult:
    """Effectue un test du chi-deux d'indépendance entre deux variables catégorielles."""
    if var1 not in df.columns or var2 not in df.columns:
        raise ValueError("Une des colonnes n'existe pas dans le DataFrame.")
    if not _is_categorical(df[var1]) or not _is_categorical(df[var2]):
        raise ValueError("Les deux variables doivent être catégorielles.")

    df_clean = df[[var1, var2]].dropna()
    warnings, preprocessing = [], []
    if len(df_clean) < len(df):
        warnings.append(f"{len(df) - len(df_clean)} observation(s) avec valeurs manquantes supprimée(s).")
        preprocessing.append("dropna")

    contingency_table = pd.crosstab(df_clean[var1], df_clean[var2])
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency_table, correction=False)
    
    n = contingency_table.values.sum()
    k = min(contingency_table.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * k))) if n > 0 and k > 0 else 0.0

    return AnalysisResult(
        method="chi_square", dataset_id=dataset_id, variables={"var1": var1, "var2": var2},
        sample_size=len(df_clean), preprocessing=preprocessing,
        estimates={"chi2": float(chi2), "dof": int(dof)}, standard_errors={}, confidence_intervals={},
        p_values={"chi2": float(p_value)}, model_metrics={"cramers_v": cramers_v},
        diagnostics=[], warnings=warnings, assumptions_checked=["independance_observations", "effectifs_theoriques>=5"],
        generated_code="import pandas as pd\nfrom scipy import stats\n\ncontingency_table = pd.crosstab(df['var1'], df['var2'])\nchi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)",
        execution_log=["dropna", "crosstab", "chi2_contingency", "cramers_v"], software_versions=_get_software_versions(),
    )


def run_t_test(df: pd.DataFrame, outcome: str, group: str, dataset_id: str = "unknown") -> AnalysisResult:
    """Effectue un test t de Student bilatéral entre deux groupes (variances égales)."""
    if outcome not in df.columns or group not in df.columns:
        raise ValueError("Une des colonnes n'existe pas dans le DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[outcome]):
        raise ValueError(f"La variable '{outcome}' n'est pas numérique.")

    df_clean = df[[outcome, group]].dropna()
    warnings, preprocessing = [], []
    if len(df_clean) < len(df):
        warnings.append(f"{len(df) - len(df_clean)} observation(s) avec valeurs manquantes supprimée(s).")
        preprocessing.append("dropna")

    groups = sorted(df_clean[group].unique())
    if len(groups) != 2:
        raise ValueError(f"La variable '{group}' doit avoir exactement 2 modalités, mais en a {len(groups)}.")

    group1_data = df_clean[df_clean[group] == groups[0]][outcome]
    group2_data = df_clean[df_clean[group] == groups[1]][outcome]
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=True)

    mean1, mean2 = float(group1_data.mean()), float(group2_data.mean())
    return AnalysisResult(
        method="t_test", dataset_id=dataset_id, variables={"outcome": outcome, "group": group, "groups": groups},
        sample_size=len(df_clean), preprocessing=preprocessing,
        estimates={f"mean_{groups[0]}": mean1, f"mean_{groups[1]}": mean2, "diff": mean1 - mean2},
        standard_errors={}, confidence_intervals={}, p_values={"t_test": float(p_value)},
        model_metrics={"t_statistic": float(t_stat), "df": float(len(df_clean) - 2)},
        diagnostics=[], warnings=warnings, assumptions_checked=["normalite_groupes", "homoscedasticite"],
        generated_code="import pandas as pd\nfrom scipy import stats\n\ngroup1 = df[df['group'] == 'g1']['outcome']\ngroup2 = df[df['group'] == 'g2']['outcome']\nt_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)",
        execution_log=["dropna", "groupby", "ttest_ind"], software_versions=_get_software_versions(),
    )


def run_anova(df: pd.DataFrame, outcome: str, group: str, dataset_id: str = "unknown") -> AnalysisResult:
    """Effectue une analyse de variance (ANOVA) à un facteur."""
    if outcome not in df.columns or group not in df.columns:
        raise ValueError("Une des colonnes n'existe pas dans le DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[outcome]):
        raise ValueError(f"La variable '{outcome}' n'est pas numérique.")

    df_clean = df[[outcome, group]].dropna()
    warnings, preprocessing = [], []
    if len(df_clean) < len(df):
        warnings.append(f"{len(df) - len(df_clean)} observation(s) avec valeurs manquantes supprimée(s).")
        preprocessing.append("dropna")

    groups = sorted(df_clean[group].unique())
    if len(groups) < 2:
        raise ValueError(f"La variable '{group}' doit avoir au moins 2 modalités, mais en a {len(groups)}.")

    group_data = [df_clean[df_clean[group] == g][outcome].values for g in groups]
    f_stat, p_value = stats.f_oneway(*group_data)

    grand_mean = df_clean[outcome].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in [df_clean[df_clean[group] == grp][outcome] for grp in groups])
    ss_total = ((df_clean[outcome] - grand_mean) ** 2).sum()
    eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0.0

    return AnalysisResult(
        method="anova", dataset_id=dataset_id, variables={"outcome": outcome, "group": group, "groups": groups},
        sample_size=len(df_clean), preprocessing=preprocessing,
        estimates={"f_statistic": float(f_stat), "df_between": len(groups) - 1, "df_within": len(df_clean) - len(groups)},
        standard_errors={}, confidence_intervals={}, p_values={"anova": float(p_value)},
        model_metrics={"eta_squared": eta_squared}, diagnostics=[], warnings=warnings,
        assumptions_checked=["normalite_groupes", "homoscedasticite", "independance"],
        generated_code="import pandas as pd\nfrom scipy import stats\n\ngroups = [df[df['group'] == g]['outcome'].values for g in df['group'].unique()]\nf_stat, p_value = stats.f_oneway(*groups)",
        execution_log=["dropna", "groupby", "f_oneway", "eta_squared"], software_versions=_get_software_versions(),
    )
