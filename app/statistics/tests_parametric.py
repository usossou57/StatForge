"""Tests paramétriques classiques (Phase 17 — Inférence V2).

Ce module implémente les tests paramétriques suivants :
- Test t pairé (échantillons appariés)
- Test t de Welch (variances inégales)
- Test t à un échantillon
- Test de proportion (z ou exact)
- Test exact de Fisher
- Test de variances (Levene, Bartlett, Fligner-Killeen)

Tous les tests retournent un AnalysisResult conforme aux contrats V1 §7 et V2 §6.2.
Les diagnostics pertinents sont appelés et placés dans result.diagnostics.
Chaque méthode est enregistrée dans le registre (V2 §7.1) avec son gabarit
de code reproductible (V2 §6.2).
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from app.diagnostics.checks_inference import (
    check_expected_counts,
    check_group_sizes,
)
from app.diagnostics.models import DiagnosticFinding, Severity
from app.errors import InvalidMethodInputError
from app.results.models import AnalysisResult
from app.statistics.registry import MethodSpec, register_method

# ============================================================================
# Helpers internes
# ============================================================================


def _drop_missing_pairs(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, int]:
    """Supprime les lignes avec des valeurs manquantes sur les colonnes données.

    Args:
        df: DataFrame d'entrée.
        columns: Liste des colonnes à vérifier.

    Returns:
        Tuple (df_nettoyé, n_lignes_supprimées).
    """
    n_before = len(df)
    df_clean = df.dropna(subset=columns)
    n_dropped = n_before - len(df_clean)
    return df_clean, n_dropped


def _get_software_versions() -> dict:
    """Retourne les versions des bibliothèques utilisées."""
    import scipy
    import statsmodels

    return {
        "python": "3.11+",
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


# ============================================================================
# Test t pairé
# ============================================================================


def run_paired_t_test(
    df: pd.DataFrame,
    var_before: str,
    var_after: str,
) -> AnalysisResult:
    """Test t pairé pour comparer deux mesures appariées.

    Args:
        df: DataFrame contenant les données.
        var_before: Nom de la colonne contenant les mesures avant.
        var_after: Nom de la colonne contenant les mesures après.

    Returns:
        AnalysisResult avec les résultats du test.

    Raises:
        InvalidMethodInputError: Si les colonnes n'existent pas ou si moins
            de 2 paires complètes.

    Note:
        Hypothèses : différences normalement distribuées, données appariées.
        Référence Python : scipy.stats.ttest_rel
        Référence R : t.test(..., paired=TRUE)
    """
    if var_before not in df.columns or var_after not in df.columns:
        raise InvalidMethodInputError(
            f"Les colonnes '{var_before}' et '{var_after}' doivent exister "
            f"dans le DataFrame."
        )

    # Suppression des paires incomplètes
    df_clean, n_dropped = _drop_missing_pairs(df, [var_before, var_after])

    if len(df_clean) < 2:
        raise InvalidMethodInputError(
            f"Au moins 2 paires complètes sont nécessaires pour un test t "
            f"pairé. Seulement {len(df_clean)} paire(s) disponible(s) après "
            f"suppression des valeurs manquantes."
        )

    before = df_clean[var_before].values
    after = df_clean[var_after].values

    # Test t pairé
    statistic, p_value = stats.ttest_rel(before, after)

    # Taille d'effet : d de Cohen pour échantillons appariés
    diff = after - before   # type: ignore[operator]
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    # IC 95% pour la différence
    se_diff = std_diff / np.sqrt(len(diff))
    ci_low, ci_high = stats.t.interval(
        0.95, df=len(diff) - 1, loc=mean_diff, scale=se_diff
    )

    # Diagnostics
    diagnostics: list[DiagnosticFinding] = []
    diagnostics.extend(check_group_sizes([diff], min_size=10))  # type: ignore[list-item]

    # Test de normalité des différences (Shapiro-Wilk si n < 5000)
    if len(diff) >= 3 and len(diff) <= 5000:
        _, p_normality = stats.shapiro(diff)
        if p_normality < 0.05:
            diagnostics.append(
                DiagnosticFinding(
                    code="NON_NORMAL_DISTRIBUTION",
                    severity=Severity.WARNING,
                    message=(
                        f"Les différences entre les mesures ne suivent pas "
                        f"une distribution normale (test de Shapiro-Wilk, "
                        f"p={p_normality:.4f}). Le test t pairé peut ne pas "
                        f"être approprié. Envisagez le test de Wilcoxon "
                        f"apparié."
                    ),
                    details={
                        "test": "shapiro",
                        "p_value": float(p_normality),
                    },
                )
            )

    # Code reproductible
    generated_code = f"""# Test t pairé
import pandas as pd
from scipy import stats

df = pd.read_csv("votre_fichier.csv")
df_clean = df.dropna(subset=["{var_before}", "{var_after}"])
before = df_clean["{var_before}"].values
after = df_clean["{var_after}"].values
statistic, p_value = stats.ttest_rel(before, after)
print(f"t = {{statistic:.4f}}, p = {{p_value:.4f}}")
"""

    return AnalysisResult(
        method="paired_t_test",
        dataset_id="",
        variables={"var_before": var_before, "var_after": var_after},
        sample_size=len(df_clean),
        preprocessing=[
            f"Suppression des paires incomplètes : {n_dropped} ligne(s) retirée(s)"
        ],
        estimates={
            "statistic": float(statistic),
            "mean_difference": float(mean_diff),
            "cohens_d": float(cohens_d),
        },
        standard_errors={"mean_difference": float(se_diff)},
        confidence_intervals={"mean_difference": [float(ci_low), float(ci_high)]},
        p_values={"overall": float(p_value)},
        model_metrics={
            "method_family": "inference",
            "effect_size": {"cohens_d": float(cohens_d)},
            "n_obs_used": len(df_clean),
            "n_obs_dropped": n_dropped,
        },
        diagnostics=diagnostics,
        warnings=[],
        assumptions_checked=[
            "Différences normalement distribuées",
            "Données appariées",
        ],
        generated_code=generated_code,
        execution_log=["Test t pairé exécuté avec succès"],
        software_versions=_get_software_versions(),
    )


# ============================================================================
# Test t de Welch
# ============================================================================


def run_welch_t_test(
    df: pd.DataFrame,
    outcome: str,
    group: str,
) -> AnalysisResult:
    """Test t de Welch pour comparer deux groupes indépendants (variances inégales).

    Args:
        df: DataFrame contenant les données.
        outcome: Nom de la colonne numérique (variable dépendante).
        group: Nom de la colonne catégorielle à 2 modalités (variable indépendante).

    Returns:
        AnalysisResult avec les résultats du test.

    Raises:
        InvalidMethodInputError: Si le groupe n'a pas exactement 2 modalités
            ou si moins de 2 observations par groupe.

    Note:
        Hypothèses : indépendance des observations, normalité dans chaque groupe.
        Ne suppose pas l'égalité des variances (contrairement au test t de Student).
        Référence Python : scipy.stats.ttest_ind(equal_var=False)
        Référence R : t.test(..., var.equal=FALSE)
    """
    if outcome not in df.columns or group not in df.columns:
        raise InvalidMethodInputError(
            f"Les colonnes '{outcome}' et '{group}' doivent exister dans le DataFrame."
        )

    # Suppression des valeurs manquantes
    df_clean, n_dropped = _drop_missing_pairs(df, [outcome, group])

    # Vérification que le groupe a exactement 2 modalités
    unique_groups = df_clean[group].dropna().unique()
    if len(unique_groups) != 2:
        raise InvalidMethodInputError(
            f"La variable de groupe '{group}' doit avoir exactement 2 modalités. "
            f"Trouvé : {len(unique_groups)} modalité(s)."
        )

    # Séparation en deux groupes
    group1_name, group2_name = sorted(unique_groups)
    group1 = df_clean[df_clean[group] == group1_name][outcome].dropna().values
    group2 = df_clean[df_clean[group] == group2_name][outcome].dropna().values

    if len(group1) < 2 or len(group2) < 2:
        raise InvalidMethodInputError(
            f"Au moins 2 observations sont nécessaires par groupe. "
            f"Groupe '{group1_name}' : {len(group1)}, Groupe '{group2_name}' : {len(group2)}."
        )

    # Test t de Welch
    statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)

    # Tailles d'effet
    mean1, mean2 = group1.mean(), group2.mean()
    std1, std2 = group1.std(ddof=1), group2.std(ddof=1)
    n1, n2 = len(group1), len(group2)

    # d de Cohen
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

    # IC 95% pour la différence de moyennes
    se_diff = np.sqrt(std1**2 / n1 + std2**2 / n2)
    df_welch = (std1**2 / n1 + std2**2 / n2) ** 2 / (
        (std1**2 / n1) ** 2 / (n1 - 1) + (std2**2 / n2) ** 2 / (n2 - 1)
    )
    ci_low, ci_high = stats.t.interval(
        0.95, df=df_welch, loc=mean1 - mean2, scale=se_diff
    )

    # Diagnostics
    diagnostics: list[DiagnosticFinding] = []
    diagnostics.extend(check_group_sizes([group1, group2], min_size=10))

    # Code reproductible
    generated_code = f"""# Test t de Welch
import pandas as pd
from scipy import stats

df = pd.read_csv("votre_fichier.csv")
df_clean = df.dropna(subset=["{outcome}", "{group}"])
group1 = df_clean[df_clean["{group}"] == "{group1_name}"]["{outcome}"].values
group2 = df_clean[df_clean["{group}"] == "{group2_name}"]["{outcome}"].values
statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
print(f"t = {{statistic:.4f}}, p = {{p_value:.4f}}")
"""

    return AnalysisResult(
        method="welch_t_test",
        dataset_id="",
        variables={"outcome": outcome, "group": group},
        sample_size=len(df_clean),
        preprocessing=[
            f"Suppression des valeurs manquantes : {n_dropped} ligne(s) retirée(s)"
        ],
        estimates={
            "statistic": float(statistic),
            "mean_difference": float(mean1 - mean2),
            "mean_group1": float(mean1),
            "mean_group2": float(mean2),
            "cohens_d": float(cohens_d),
        },
        standard_errors={"mean_difference": float(se_diff)},
        confidence_intervals={"mean_difference": [float(ci_low), float(ci_high)]},
        p_values={"overall": float(p_value)},
        model_metrics={
            "method_family": "inference",
            "effect_size": {"cohens_d": float(cohens_d)},
            "n_obs_used": len(df_clean),
            "n_obs_dropped": n_dropped,
            "group_sizes": {
                group1_name: n1,
                group2_name: n2,
            },
        },
        diagnostics=diagnostics,
        warnings=[],
        assumptions_checked=[
            "Indépendance des observations",
            "Normalité dans chaque groupe",
            "Ne suppose pas l'égalité des variances",
        ],
        generated_code=generated_code,
        execution_log=["Test t de Welch exécuté avec succès"],
        software_versions=_get_software_versions(),
    )


# ============================================================================
# Test t à un échantillon
# ============================================================================


def run_one_sample_t_test(
    df: pd.DataFrame,
    column: str,
    mu0: float,
) -> AnalysisResult:
    """Test t à un échantillon pour comparer la moyenne à une valeur de référence.

    Args:
        df: DataFrame contenant les données.
        column: Nom de la colonne numérique à tester.
        mu0: Valeur de référence pour la moyenne (hypothèse nulle).

    Returns:
        AnalysisResult avec les résultats du test.

    Raises:
        InvalidMethodInputError: Si la colonne n'existe pas ou si moins de
            2 observations valides.

    Note:
        Hypothèses : échantillon normalement distribué.
        Référence Python : scipy.stats.ttest_1samp
        Référence R : t.test(..., mu=mu0)
    """
    if column not in df.columns:
        raise InvalidMethodInputError(
            f"La colonne '{column}' doit exister dans le DataFrame."
        )

    # Suppression des valeurs manquantes
    values = df[column].dropna().values

    if len(values) < 2:
        raise InvalidMethodInputError(
            f"Au moins 2 observations valides sont nécessaires. "
            f"Seulement {len(values)} observation(s) disponible(s)."
        )

    # Test t à un échantillon
    statistic, p_value = stats.ttest_1samp(values, mu0)

    # Statistiques descriptives
    mean_val = values.mean()    # type: ignore[union-attr]
    std_val = values.std(ddof=1)    # type: ignore[union-attr]
    se_val = std_val / np.sqrt(len(values))

    # d de Cohen
    cohens_d = (mean_val - mu0) / std_val if std_val > 0 else 0.0

    # IC 95% pour la moyenne
    ci_low, ci_high = stats.t.interval(
        0.95, df=len(values) - 1, loc=mean_val, scale=se_val
    )

    # Diagnostics
    diagnostics: list[DiagnosticFinding] = []
    diagnostics.extend(check_group_sizes([values], min_size=10))    # type: ignore[list-item]

    # Test de normalité (Shapiro-Wilk si n < 5000)
    if len(values) >= 3 and len(values) <= 5000:
        _, p_normality = stats.shapiro(values)
        if p_normality < 0.05:
            diagnostics.append(
                DiagnosticFinding(
                    code="NON_NORMAL_DISTRIBUTION",
                    severity=Severity.WARNING,
                    message=(
                        f"La variable ne suit pas une distribution normale "
                        f"(test de Shapiro-Wilk, p={p_normality:.4f}). "
                        f"Le test t à un échantillon peut ne pas être approprié."
                    ),
                    details={
                        "test": "shapiro",
                        "p_value": float(p_normality),
                    },
                )
            )

    # Code reproductible
    generated_code = f"""# Test t à un échantillon
import pandas as pd
from scipy import stats

df = pd.read_csv("votre_fichier.csv")
values = df["{column}"].dropna().values
statistic, p_value = stats.ttest_1samp(values, {mu0})
print(f"t = {{statistic:.4f}}, p = {{p_value:.4f}}")
"""

    return AnalysisResult(
        method="one_sample_t_test",
        dataset_id="",
        variables={"column": column, "mu0": mu0},
        sample_size=len(values),
        preprocessing=[
            f"Suppression des valeurs manquantes : {len(df) - len(values)} ligne(s) retirée(s)"
        ],
        estimates={
            "statistic": float(statistic),
            "mean": float(mean_val),
            "mu0": float(mu0),
            "mean_difference": float(mean_val - mu0),
            "cohens_d": float(cohens_d),
        },
        standard_errors={"mean": float(se_val)},
        confidence_intervals={"mean": [float(ci_low), float(ci_high)]},
        p_values={"overall": float(p_value)},
        model_metrics={
            "method_family": "inference",
            "effect_size": {"cohens_d": float(cohens_d)},
            "n_obs_used": len(values),
            "n_obs_dropped": len(df) - len(values),
        },
        diagnostics=diagnostics,
        warnings=[],
        assumptions_checked=[
            "Échantillon normalement distribué",
        ],
        generated_code=generated_code,
        execution_log=["Test t à un échantillon exécuté avec succès"],
        software_versions=_get_software_versions(),
    )


# ============================================================================
# Test de proportion
# ============================================================================


def run_proportion_test(
    df: pd.DataFrame,
    column: str,
    success_value: Any,
    p0: float = 0.5,
    group: str | None = None,
) -> AnalysisResult:
    """Test de proportion (z ou exact) pour comparer une proportion à une valeur de référence.

    Args:
        df: DataFrame contenant les données.
        column: Nom de la colonne à tester.
        success_value: Valeur considérée comme un "succès".
        p0: Proportion de référence sous l'hypothèse nulle (défaut : 0.5).
        group: Nom optionnel d'une colonne de groupe pour comparer deux proportions.

    Returns:
        AnalysisResult avec les résultats du test.

    Raises:
        InvalidMethodInputError: Si la colonne n'existe pas ou si les effectifs
            sont insuffisants.

    Note:
        Si group est None : test à un échantillon (z ou exact).
        Si group est fourni : test de comparaison de deux proportions (z).
        Référence Python : statsmodels.stats.proportion
        Référence R : prop.test, binom.test
    """
    if column not in df.columns:
        raise InvalidMethodInputError(
            f"La colonne '{column}' doit exister dans le DataFrame."
        )

    if group is not None and group not in df.columns:
        raise InvalidMethodInputError(
            f"La colonne de groupe '{group}' doit exister dans le DataFrame."
        )

    # Suppression des valeurs manquantes
    columns_to_check = [column] + ([group] if group else [])
    df_clean, n_dropped = _drop_missing_pairs(df, columns_to_check)

    if group is None:
        # Test à un échantillon
        successes = (df_clean[column] == success_value).sum()
        n_total = len(df_clean)

        if n_total == 0:
            raise InvalidMethodInputError(
                "Aucune observation valide après suppression des valeurs manquantes."
            )

        # Test exact binomial si n < 30, sinon test z
        if n_total < 30:
            # CORRECTION : stats.binomtest retourne un objet, on extrait la p-value
            binom_res = stats.binomtest(successes, n_total, p0)
            p_value = float(binom_res.pvalue)
            statistic = float(successes)
            test_type = "exact"
        else:
            # Test z
            statistic, p_value = proportions_ztest(successes, n_total, value=p0)
            test_type = "z"

        # Proportion observée
        p_hat = successes / n_total

        # IC 95% (méthode de Wilson)
        ci_low, ci_high = proportion_confint(
            successes, n_total, alpha=0.05, method="wilson"
        )

        # Diagnostics
        diagnostics: list[DiagnosticFinding] = []
        if n_total < 10:
            diagnostics.extend(check_group_sizes([[1] * n_total], min_size=10)) # type: ignore[list-item]

        # Code reproductible
        generated_code = f"""# Test de proportion
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

df = pd.read_csv("votre_fichier.csv")
df_clean = df.dropna(subset=["{column}"])
successes = (df_clean["{column}"] == {success_value!r}).sum()
n_total = len(df_clean)
statistic, p_value = proportions_ztest(successes, n_total, value={p0})
ci_low, ci_high = proportion_confint(successes, n_total, alpha=0.05, method="wilson")
print(f"Proportion = {{successes/n_total:.4f}}, p = {{p_value:.4f}}")
"""

        return AnalysisResult(
            method="proportion_test",
            dataset_id="",
            variables={"column": column, "success_value": str(success_value), "p0": p0},
            sample_size=n_total,
            preprocessing=[
                f"Suppression des valeurs manquantes : {n_dropped} ligne(s) retirée(s)"
            ],
            estimates={
                "statistic": float(statistic),
                "proportion": float(p_hat),
                "successes": int(successes),
                "n_total": int(n_total),
            },
            standard_errors={},
            confidence_intervals={"proportion": [float(ci_low), float(ci_high)]},
            p_values={"overall": float(p_value)},
            model_metrics={
                "method_family": "inference",
                "test_type": test_type,
                "n_obs_used": n_total,
                "n_obs_dropped": n_dropped,
            },
            diagnostics=diagnostics,
            warnings=[],
            assumptions_checked=[
                "Observations indépendantes",
                "Effectifs suffisants pour l'approximation normale"
                if test_type == "z"
                else "Test exact",
            ],
            generated_code=generated_code,
            execution_log=[f"Test de proportion ({test_type}) exécuté avec succès"],
            software_versions=_get_software_versions(),
        )

    else:
        # Test de comparaison de deux proportions
        group_values = df_clean[group].dropna().unique()
        if len(group_values) != 2:
            raise InvalidMethodInputError(
                f"La variable de groupe '{group}' doit avoir exactement 2 modalités. "
                f"Trouvé : {len(group_values)} modalité(s)."
            )

        group1_name, group2_name = sorted(group_values)
        successes1 = (
            df_clean[df_clean[group] == group1_name][column] == success_value
        ).sum()
        n1 = (df_clean[group] == group1_name).sum()
        successes2 = (
            df_clean[df_clean[group] == group2_name][column] == success_value
        ).sum()
        n2 = (df_clean[group] == group2_name).sum()

        if n1 == 0 or n2 == 0:
            raise InvalidMethodInputError(
                "Chaque groupe doit contenir au moins une observation."
            )

        # Test z pour deux proportions
        count = np.array([successes1, successes2])
        nobs = np.array([n1, n2])
        statistic, p_value = proportions_ztest(count, nobs)

        p1_hat = successes1 / n1
        p2_hat = successes2 / n2

        # IC 95% pour chaque proportion
        ci1_low, ci1_high = proportion_confint(
            successes1, n1, alpha=0.05, method="wilson"
        )
        ci2_low, ci2_high = proportion_confint(
            successes2, n2, alpha=0.05, method="wilson"
        )

        # Diagnostics
        diagnostics = []
        diagnostics.extend(check_group_sizes([[1] * n1, [1] * n2], min_size=10))    # type: ignore[list-item]

        # Code reproductible
        generated_code = f"""# Test de comparaison de deux proportions
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

df = pd.read_csv("votre_fichier.csv")
df_clean = df.dropna(subset=["{column}", "{group}"])
successes1 = (df_clean[df_clean["{group}"] == "{group1_name}"]["{column}"] == {success_value!r}).sum()
n1 = (df_clean["{group}"] == "{group1_name}").sum()
successes2 = (df_clean[df_clean["{group}"] == "{group2_name}"]["{column}"] == {success_value!r}).sum()
n2 = (df_clean["{group}"] == "{group2_name}").sum()
count = np.array([successes1, successes2])
nobs = np.array([n1, n2])
statistic, p_value = proportions_ztest(count, nobs)
print(f"z = {{statistic:.4f}}, p = {{p_value:.4f}}")
"""

        return AnalysisResult(
            method="proportion_test_two_sample",
            dataset_id="",
            variables={
                "column": column,
                "success_value": str(success_value),
                "group": group,
            },
            sample_size=len(df_clean),
            preprocessing=[
                f"Suppression des valeurs manquantes : {n_dropped} ligne(s) retirée(s)"
            ],
            estimates={
                "statistic": float(statistic),
                "proportion_group1": float(p1_hat),
                "proportion_group2": float(p2_hat),
                "proportion_difference": float(p1_hat - p2_hat),
            },
            standard_errors={},
            confidence_intervals={
                "proportion_group1": [float(ci1_low), float(ci1_high)],
                "proportion_group2": [float(ci2_low), float(ci2_high)],
            },
            p_values={"overall": float(p_value)},
            model_metrics={
                "method_family": "inference",
                "test_type": "z",
                "n_obs_used": len(df_clean),
                "n_obs_dropped": n_dropped,
                "group_sizes": {
                    group1_name: n1,
                    group2_name: n2,
                },
            },
            diagnostics=diagnostics,
            warnings=[],
            assumptions_checked=[
                "Observations indépendantes",
                "Effectifs suffisants pour l'approximation normale",
            ],
            generated_code=generated_code,
            execution_log=[
                "Test de comparaison de deux proportions exécuté avec succès"
            ],
            software_versions=_get_software_versions(),
        )


# ============================================================================
# Test exact de Fisher
# ============================================================================


def run_fisher_exact(
    df: pd.DataFrame,
    var1: str,
    var2: str,
) -> AnalysisResult:
    """Test exact de Fisher pour tableaux de contingence 2x2.

    Args:
        df: DataFrame contenant les données.
        var1: Nom de la première variable catégorielle.
        var2: Nom de la deuxième variable catégorielle.

    Returns:
        AnalysisResult avec les résultats du test.

    Raises:
        InvalidMethodInputError: Si les colonnes n'existent pas ou si le
            tableau de contingence n'est pas 2x2.

    Note:
        Hypothèses : pas d'hypothèse de distribution, adapté aux petits effectifs.
        Référence Python : scipy.stats.fisher_exact
        Référence R : fisher.test
    """
    if var1 not in df.columns or var2 not in df.columns:
        raise InvalidMethodInputError(
            f"Les colonnes '{var1}' et '{var2}' doivent exister dans le DataFrame."
        )

    # Suppression des valeurs manquantes
    df_clean, n_dropped = _drop_missing_pairs(df, [var1, var2])

    if len(df_clean) == 0:
        raise InvalidMethodInputError(
            "Aucune observation valide après suppression des valeurs manquantes."
        )

    # Tableau de contingence
    contingency = pd.crosstab(df_clean[var1], df_clean[var2])

    if contingency.shape != (2, 2):
        raise InvalidMethodInputError(
            f"Le test exact de Fisher nécessite un tableau de contingence 2x2. "
            f"Trouvé : {contingency.shape[0]}x{contingency.shape[1]}. "
            f"Pour des tableaux plus grands, utilisez le test du chi²."
        )

    # Test exact de Fisher
    odds_ratio, p_value = stats.fisher_exact(contingency.values)

    # Diagnostics
    diagnostics: list[DiagnosticFinding] = []
    diagnostics.extend(check_expected_counts(contingency, threshold=5))

    # Code reproductible
    generated_code = f"""# Test exact de Fisher
import pandas as pd
from scipy import stats

df = pd.read_csv("votre_fichier.csv")
df_clean = df.dropna(subset=["{var1}", "{var2}"])
contingency = pd.crosstab(df_clean["{var1}"], df_clean["{var2}"])
odds_ratio, p_value = stats.fisher_exact(contingency.values)
print(f"OR = {{odds_ratio:.4f}}, p = {{p_value:.4f}}")
"""

    return AnalysisResult(
        method="fisher_exact",
        dataset_id="",
        variables={"var1": var1, "var2": var2},
        sample_size=len(df_clean),
        preprocessing=[
            f"Suppression des valeurs manquantes : {n_dropped} ligne(s) retirée(s)"
        ],
        estimates={
            "odds_ratio": float(odds_ratio),
        },
        standard_errors={},
        confidence_intervals={},
        p_values={"overall": float(p_value)},
        model_metrics={
            "method_family": "inference",
            "contingency_table": contingency.to_dict(),
            "n_obs_used": len(df_clean),
            "n_obs_dropped": n_dropped,
        },
        diagnostics=diagnostics,
        warnings=[],
        assumptions_checked=[
            "Aucune hypothèse de distribution",
            "Adapté aux petits effectifs",
        ],
        generated_code=generated_code,
        execution_log=["Test exact de Fisher exécuté avec succès"],
        software_versions=_get_software_versions(),
    )


# ============================================================================
# Test de variances
# ============================================================================


def run_variance_test(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    method: str = "levene",
) -> AnalysisResult:
    """Test d'égalité des variances entre groupes.

    Args:
        df: DataFrame contenant les données.
        outcome: Nom de la colonne numérique (variable dépendante).
        group: Nom de la colonne catégorielle (variable indépendante).
        method: Méthode de test ("levene", "bartlett", ou "fligner").

    Returns:
        AnalysisResult avec les résultats du test.

    Raises:
        InvalidMethodInputError: Si les colonnes n'existent pas, si la méthode
            est invalide, ou si moins de 2 groupes.

    Note:
        - Levene : plus robuste aux non-normalités (recommandé par défaut)
        - Bartlett : plus puissant si normalité, mais sensible aux écarts
        - Fligner-Killeen : robuste, basé sur les rangs
        Référence Python : scipy.stats.levene/bartlett/fligner
        Référence R : var.test (Bartlett), leveneTest (car)
    """
    if outcome not in df.columns or group not in df.columns:
        raise InvalidMethodInputError(
            f"Les colonnes '{outcome}' et '{group}' doivent exister dans le DataFrame."
        )

    if method not in ["levene", "bartlett", "fligner"]:
        raise InvalidMethodInputError(
            f"Méthode invalide : '{method}'. Choisir parmi 'levene', 'bartlett', 'fligner'."
        )

    # Suppression des valeurs manquantes
    df_clean, n_dropped = _drop_missing_pairs(df, [outcome, group])

    # Vérification que le groupe a au moins 2 modalités
    unique_groups = df_clean[group].dropna().unique()
    if len(unique_groups) < 2:
        raise InvalidMethodInputError(
            f"La variable de groupe '{group}' doit avoir au moins 2 modalités. "
            f"Trouvé : {len(unique_groups)} modalité(s)."
        )

    # Séparation en groupes
    groups = [
        df_clean[df_clean[group] == g][outcome].dropna().values
        for g in sorted(unique_groups)
    ]

    # Vérification effectifs minimaux
    for i, g in enumerate(groups):
        if len(g) < 2:
            raise InvalidMethodInputError(
                f"Le groupe {i + 1} ne contient que {len(g)} observation(s). "
                f"Au moins 2 observations sont nécessaires par groupe."
            )

    # Test selon la méthode choisie
    if method == "levene":
        statistic, p_value = stats.levene(*groups, center="median")
        test_name = "Levene"
    elif method == "bartlett":
        statistic, p_value = stats.bartlett(*groups)
        test_name = "Bartlett"
    else:  # fligner
        statistic, p_value = stats.fligner(*groups)
        test_name = "Fligner-Killeen"

    # Diagnostics
    diagnostics: list[DiagnosticFinding] = []
    diagnostics.extend(check_group_sizes(groups, min_size=10))

    # Code reproductible
    generated_code = f"""# Test de variances ({test_name})
import pandas as pd
from scipy import stats

df = pd.read_csv("votre_fichier.csv")
df_clean = df.dropna(subset=["{outcome}", "{group}"])
groups = [df_clean[df_clean["{group}"] == g]["{outcome}"].values for g in sorted(df_clean["{group}"].unique())]
statistic, p_value = stats.{method}(*groups)
print(f"F = {{statistic:.4f}}, p = {{p_value:.4f}}")
"""

    return AnalysisResult(
        method=f"variance_test_{method}",
        dataset_id="",
        variables={"outcome": outcome, "group": group, "method": method},
        sample_size=len(df_clean),
        preprocessing=[
            f"Suppression des valeurs manquantes : {n_dropped} ligne(s) retirée(s)"
        ],
        estimates={
            "statistic": float(statistic),
        },
        standard_errors={},
        confidence_intervals={},
        p_values={"overall": float(p_value)},
        model_metrics={
            "method_family": "inference",
            "test_name": test_name,
            "n_groups": len(groups),
            "group_sizes": {
                str(g): len(grp) for g, grp in zip(sorted(unique_groups), groups)
            },
            "n_obs_used": len(df_clean),
            "n_obs_dropped": n_dropped,
        },
        diagnostics=diagnostics,
        warnings=[],
        assumptions_checked=[
            "Indépendance des observations",
            "Normalité" if method == "bartlett" else "Plus robuste aux non-normalités",
        ],
        generated_code=generated_code,
        execution_log=[f"Test de variances ({test_name}) exécuté avec succès"],
        software_versions=_get_software_versions(),
    )


# ============================================================================
# Enregistrement dans le registre
# ============================================================================


# MethodSpec pour chaque test
_PAIRED_T_TEST_SPEC = MethodSpec(
    method_id="paired_t_test",
    label_fr="Test t pairé",
    family="inference",
    engine="python",
    outcome_types=["continuous"],
    predictor_types=["continuous"],
    min_sample_size=2,
    assumptions=[
        "Différences normalement distribuées",
        "Données appariées",
    ],
    reference_impl="scipy.stats.ttest_rel",
    r_reference="t.test(..., paired=TRUE)",
    phase=17,
)

_WELCH_T_TEST_SPEC = MethodSpec(
    method_id="welch_t_test",
    label_fr="Test t de Welch",
    family="inference",
    engine="python",
    outcome_types=["continuous"],
    predictor_types=["binary"],
    min_sample_size=4,
    assumptions=[
        "Indépendance des observations",
        "Normalité dans chaque groupe",
        "Ne suppose pas l'égalité des variances",
    ],
    reference_impl="scipy.stats.ttest_ind(equal_var=False)",
    r_reference="t.test(..., var.equal=FALSE)",
    phase=17,
)

_ONE_SAMPLE_T_TEST_SPEC = MethodSpec(
    method_id="one_sample_t_test",
    label_fr="Test t à un échantillon",
    family="inference",
    engine="python",
    outcome_types=["continuous"],
    predictor_types=[],
    min_sample_size=2,
    assumptions=[
        "Échantillon normalement distribué",
    ],
    reference_impl="scipy.stats.ttest_1samp",
    r_reference="t.test(..., mu=mu0)",
    phase=17,
)

_PROPORTION_TEST_SPEC = MethodSpec(
    method_id="proportion_test",
    label_fr="Test de proportion",
    family="inference",
    engine="python",
    outcome_types=["binary", "categorical"],
    predictor_types=["binary"],
    min_sample_size=1,
    assumptions=[
        "Observations indépendantes",
        "Effectifs suffisants pour l'approximation normale",
    ],
    reference_impl="statsmodels.stats.proportion.proportions_ztest",
    r_reference="prop.test, binom.test",
    phase=17,
)

_FISHER_EXACT_SPEC = MethodSpec(
    method_id="fisher_exact",
    label_fr="Test exact de Fisher",
    family="inference",
    engine="python",
    outcome_types=["binary", "categorical"],
    predictor_types=["binary", "categorical"],
    min_sample_size=1,
    assumptions=[
        "Aucune hypothèse de distribution",
        "Adapté aux petits effectifs",
    ],
    reference_impl="scipy.stats.fisher_exact",
    r_reference="fisher.test",
    phase=17,
)

_VARIANCE_TEST_SPEC = MethodSpec(
    method_id="variance_test",
    label_fr="Test d'égalité des variances",
    family="inference",
    engine="python",
    outcome_types=["continuous"],
    predictor_types=["categorical"],
    min_sample_size=4,
    assumptions=[
        "Indépendance des observations",
        "Normalité (Bartlett) ou plus robuste (Levene, Fligner)",
    ],
    reference_impl="scipy.stats.levene/bartlett/fligner",
    r_reference="var.test, car::leveneTest",
    phase=17,
)

# Enregistrement des méthodes
register_method(_PAIRED_T_TEST_SPEC, run_paired_t_test)
register_method(_WELCH_T_TEST_SPEC, run_welch_t_test)
register_method(_ONE_SAMPLE_T_TEST_SPEC, run_one_sample_t_test)
register_method(_PROPORTION_TEST_SPEC, run_proportion_test)
register_method(_FISHER_EXACT_SPEC, run_fisher_exact)
register_method(_VARIANCE_TEST_SPEC, run_variance_test)

# Enregistrement des gabarits de code reproductible
# (Le code est déjà inclus dans chaque AnalysisResult.generated_code)
# register_codegen est appelé implicitement via generated_code dans chaque fonction
