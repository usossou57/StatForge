"""Tests non paramétriques classiques (Phase 17 — Inférence V2).

Ce module implémente les tests non paramétriques suivants :
- Test de Mann-Whitney U (comparaison de deux groupes indépendants)
- Test de Wilcoxon signed-rank (comparaison de deux mesures appariées)
- Test de Kruskal-Wallis (comparaison de k groupes indépendants)
- Test de Friedman (comparaison de k mesures appariées)
- Test de Spearman (corrélation de rangs)
- Test de Kendall (corrélation de rangs)

Tous les tests retournent un AnalysisResult conforme aux contrats V1 §7 et V2 §6.2.
Les diagnostics pertinents sont appelés et placés dans result.diagnostics.
Chaque méthode est enregistrée dans le registre (V2 §7.1) avec son gabarit
de code reproductible (V2 §6.2).
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.diagnostics.checks_inference import check_group_sizes
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
    """Supprime les lignes avec des valeurs manquantes sur les colonnes données."""
    n_before = len(df)
    df_clean = df.dropna(subset=columns)
    n_dropped = n_before - len(df_clean)
    return df_clean, n_dropped


def _get_software_versions() -> dict:
    """Retourne les versions des bibliothèques utilisées."""
    import scipy

    return {
        "python": "3.11+",
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def _check_tied_ranks(
    groups: list[np.ndarray],
    threshold: float = 0.2,
) -> list[DiagnosticFinding]:
    """Vérifie la proportion de rangs liés (ex-aequo).

    Args:
        groups: Liste de tableaux numpy, un par groupe.
        threshold: Proportion maximale de rangs liés acceptée (défaut : 20%).

    Returns:
        Liste de DiagnosticFinding. Retourne TIED_RANKS si la proportion
        de rangs liés dépasse le seuil.

    Note:
        Les tests non paramétriques basés sur les rangs peuvent perdre en
        puissance avec de nombreux ex-aequo. Source : pratique courante.
    """
    all_values = np.concatenate(groups)
    n_total = len(all_values)

    if n_total == 0:
        return []

    # Calcul des rangs avec gestion des ex-aequo
    ranks = stats.rankdata(all_values, method="average")
    unique_ranks, counts = np.unique(ranks, return_counts=True)

    # Nombre de valeurs impliquées dans des ex-aequo
    tied_values = counts[counts > 1]
    n_tied = tied_values.sum() if len(tied_values) > 0 else 0
    proportion_tied = n_tied / n_total

    findings: list[DiagnosticFinding] = []
    if proportion_tied > threshold:
        findings.append(
            DiagnosticFinding(
                code="TIED_RANKS",
                severity=Severity.WARNING,
                message=(
                    f"{proportion_tied:.1%} des observations sont des ex-aequo. "
                    f"Les tests basés sur les rangs peuvent perdre en puissance "
                    f"avec de nombreux ex-aequo."
                ),
                details={
                    "proportion_tied": float(proportion_tied),
                    "n_tied": int(n_tied),
                    "n_total": int(n_total),
                    "threshold": threshold,
                },
            )
        )

    return findings


# ============================================================================
# Test de Mann-Whitney U
# ============================================================================


def run_mann_whitney(
    df: pd.DataFrame,
    outcome: str,
    group: str,
) -> AnalysisResult:
    """Test de Mann-Whitney U pour comparer deux groupes indépendants.

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
        Hypothèses : indépendance des observations, distributions de même forme.
        Référence Python : scipy.stats.mannwhitneyu
        Référence R : stats::wilcox.test
    """
    if outcome not in df.columns or group not in df.columns:
        raise InvalidMethodInputError(
            f"Les colonnes '{outcome}' et '{group}' doivent exister dans le DataFrame."
        )

    df_clean, n_dropped = _drop_missing_pairs(df, [outcome, group])

    unique_groups = df_clean[group].dropna().unique()
    if len(unique_groups) != 2:
        raise InvalidMethodInputError(
            f"La variable de groupe '{group}' doit avoir exactement 2 modalités. "
            f"Trouvé : {len(unique_groups)} modalité(s)."
        )

    group1_name, group2_name = sorted(unique_groups)
    group1 = df_clean[df_clean[group] == group1_name][outcome].dropna().values
    group2 = df_clean[df_clean[group] == group2_name][outcome].dropna().values

    if len(group1) < 2 or len(group2) < 2:
        raise InvalidMethodInputError(
            f"Au moins 2 observations sont nécessaires par groupe. "
            f"Groupe '{group1_name}' : {len(group1)}, Groupe '{group2_name}' : {len(group2)}."
        )

    # Test de Mann-Whitney U
    statistic, p_value = stats.mannwhitneyu(
        group1, group2, use_continuity=True, alternative="two-sided"
    )

    # Taille d'effet : corrélation bisériale de rang
    n1, n2 = len(group1), len(group2)
    rank_biserial = 1 - (2 * statistic) / (n1 * n2)

    # Médianes par groupe
    median1, median2 = float(np.median(group1)), float(np.median(group2))

    # Diagnostics
    diagnostics: list[DiagnosticFinding] = []
    diagnostics.extend(check_group_sizes([group1, group2], min_size=10))
    diagnostics.extend(_check_tied_ranks([group1, group2]))

    # Code reproductible
    generated_code = f"""# Test de Mann-Whitney U
import pandas as pd
from scipy import stats

df = pd.read_csv("votre_fichier.csv")
df_clean = df.dropna(subset=["{outcome}", "{group}"])
group1 = df_clean[df_clean["{group}"] == "{group1_name}"]["{outcome}"].values
group2 = df_clean[df_clean["{group}"] == "{group2_name}"]["{outcome}"].values
statistic, p_value = stats.mannwhitneyu(group1, group2, use_continuity=True, alternative="two-sided")
print(f"U = {{statistic:.4f}}, p = {{p_value:.4f}}")
"""

    return AnalysisResult(
        method="mann_whitney",
        dataset_id="",
        variables={"outcome": outcome, "group": group},
        sample_size=int(len(df_clean)),
        preprocessing=[
            f"Suppression des valeurs manquantes : {n_dropped} ligne(s) retirée(s)"
        ],
        estimates={
            "statistic": float(statistic),
            "median_group1": float(median1),
            "median_group2": float(median2),
            "median_difference": float(median1 - median2),
        },
        standard_errors={},
        confidence_intervals={},
        p_values={"overall": float(p_value)},
        model_metrics={
            "method_family": "inference",
            "effect_size": {"rank_biserial": float(rank_biserial)},
            "n_obs_used": int(len(df_clean)),
            "n_obs_dropped": int(n_dropped),
            "group_sizes": {str(group1_name): int(n1), str(group2_name): int(n2)},
        },
        diagnostics=diagnostics,
        warnings=[],
        assumptions_checked=[
            "Indépendance des observations",
            "Distributions de même forme",
        ],
        generated_code=generated_code,
        execution_log=["Test de Mann-Whitney U exécuté avec succès"],
        software_versions=_get_software_versions(),
    )


# ============================================================================
# Test de Wilcoxon signed-rank
# ============================================================================


def run_wilcoxon_signed_rank(
    df: pd.DataFrame,
    var_before: str,
    var_after: str,
) -> AnalysisResult:
    """Test de Wilcoxon signed-rank pour comparer deux mesures appariées.

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
        Hypothèses : différences symétriquement distribuées, données appariées.
        Référence Python : scipy.stats.wilcoxon
        Référence R : stats::wilcox.test(..., paired=TRUE)
    """
    if var_before not in df.columns or var_after not in df.columns:
        raise InvalidMethodInputError(
            f"Les colonnes '{var_before}' et '{var_after}' doivent exister dans le DataFrame."
        )

    df_clean, n_dropped = _drop_missing_pairs(df, [var_before, var_after])

    if len(df_clean) < 2:
        raise InvalidMethodInputError(
            f"Au moins 2 paires complètes sont nécessaires. "
            f"Seulement {len(df_clean)} paire(s) disponible(s) après suppression."
        )

    before = df_clean[var_before].values
    after = df_clean[var_after].values

      # Test de Wilcoxon signed-rank
    # method='approx' et correction=True pour correspondre exactement à R (exact=FALSE, correct=TRUE)
    statistic, p_value = stats.wilcoxon(before, after, alternative="two-sided", method="approx", correction=True)

    # Taille d'effet : r = Z / sqrt(N)
    # Approximation de Z à partir de W (statistic)
    n = len(before)
    max_w = n * (n + 1) / 2
    z = (statistic - max_w / 2) / np.sqrt(max_w / 4)
    effect_size_r = abs(z) / np.sqrt(n)

    # Médiane des différences
    diff = after - before
    median_diff = float(np.median(diff))

    # Diagnostics
    diagnostics: list[DiagnosticFinding] = []
    diagnostics.extend(check_group_sizes([diff], min_size=10))
    diagnostics.extend(_check_tied_ranks([diff]))

    # Code reproductible
    generated_code = f"""# Test de Wilcoxon signed-rank
import pandas as pd
from scipy import stats

df = pd.read_csv("votre_fichier.csv")
df_clean = df.dropna(subset=["{var_before}", "{var_after}"])
before = df_clean["{var_before}"].values
after = df_clean["{var_after}"].values
statistic, p_value = stats.wilcoxon(before, after, alternative="two-sided")
print(f"W = {{statistic:.4f}}, p = {{p_value:.4f}}")
"""

    return AnalysisResult(
        method="wilcoxon_signed_rank",
        dataset_id="",
        variables={"var_before": var_before, "var_after": var_after},
        sample_size=int(len(df_clean)),
        preprocessing=[
            f"Suppression des paires incomplètes : {n_dropped} ligne(s) retirée(s)"
        ],
        estimates={
            "statistic": float(statistic),
            "median_difference": float(median_diff),
        },
        standard_errors={},
        confidence_intervals={},
        p_values={"overall": float(p_value)},
        model_metrics={
            "method_family": "inference",
            "effect_size": {"r": float(effect_size_r)},
            "n_obs_used": int(len(df_clean)),
            "n_obs_dropped": int(n_dropped),
        },
        diagnostics=diagnostics,
        warnings=[],
        assumptions_checked=[
            "Différences symétriquement distribuées",
            "Données appariées",
        ],
        generated_code=generated_code,
        execution_log=["Test de Wilcoxon signed-rank exécuté avec succès"],
        software_versions=_get_software_versions(),
    )


# ============================================================================
# Test de Kruskal-Wallis
# ============================================================================


def run_kruskal_wallis(
    df: pd.DataFrame,
    outcome: str,
    group: str,
) -> AnalysisResult:
    """Test de Kruskal-Wallis pour comparer k groupes indépendants.

    Args:
        df: DataFrame contenant les données.
        outcome: Nom de la colonne numérique (variable dépendante).
        group: Nom de la colonne catégorielle (variable indépendante).

    Returns:
        AnalysisResult avec les résultats du test.

    Raises:
        InvalidMethodInputError: Si le groupe n'a pas au moins 2 modalités
            ou si moins de 2 observations par groupe.

    Note:
        Hypothèses : indépendance des observations, distributions de même forme.
        Référence Python : scipy.stats.kruskal
        Référence R : stats::kruskal.test
    """
    if outcome not in df.columns or group not in df.columns:
        raise InvalidMethodInputError(
            f"Les colonnes '{outcome}' et '{group}' doivent exister dans le DataFrame."
        )

    df_clean, n_dropped = _drop_missing_pairs(df, [outcome, group])

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

    # Test de Kruskal-Wallis
    statistic, p_value = stats.kruskal(*groups)

    # Taille d'effet : eta-squared = (H - k + 1) / (n - k)
    k = len(groups)
    n = sum(len(g) for g in groups)
    eta_squared = (statistic - k + 1) / (n - k) if n > k else 0.0

    # Médianes par groupe
    medians = {str(g): float(np.median(grp)) for g, grp in zip(sorted(unique_groups), groups)}

    # Diagnostics
    diagnostics: list[DiagnosticFinding] = []
    diagnostics.extend(check_group_sizes(groups, min_size=10))
    diagnostics.extend(_check_tied_ranks(groups))

    # Code reproductible
    generated_code = f"""# Test de Kruskal-Wallis
import pandas as pd
from scipy import stats

df = pd.read_csv("votre_fichier.csv")
df_clean = df.dropna(subset=["{outcome}", "{group}"])
groups = [df_clean[df_clean["{group}"] == g]["{outcome}"].values for g in sorted(df_clean["{group}"].unique())]
statistic, p_value = stats.kruskal(*groups)
print(f"H = {{statistic:.4f}}, p = {{p_value:.4f}}")
"""

    return AnalysisResult(
        method="kruskal_wallis",
        dataset_id="",
        variables={"outcome": outcome, "group": group},
        sample_size=int(len(df_clean)),
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
            "effect_size": {"eta_squared": float(eta_squared)},
            "n_groups": int(k),
            "group_medians": medians,
            "n_obs_used": int(len(df_clean)),
            "n_obs_dropped": int(n_dropped),
            "group_sizes": {str(g): int(len(grp)) for g, grp in zip(sorted(unique_groups), groups)},
        },
        diagnostics=diagnostics,
        warnings=[],
        assumptions_checked=[
            "Indépendance des observations",
            "Distributions de même forme",
        ],
        generated_code=generated_code,
        execution_log=["Test de Kruskal-Wallis exécuté avec succès"],
        software_versions=_get_software_versions(),
    )


# ============================================================================
# Test de Friedman
# ============================================================================


def run_friedman(
    df: pd.DataFrame,
    variables: list[str],
) -> AnalysisResult:
    """Test de Friedman pour comparer k mesures appariées.

    Args:
        df: DataFrame contenant les données.
        variables: Liste des noms de colonnes à comparer (au moins 3).

    Returns:
        AnalysisResult avec les résultats du test.

    Raises:
        InvalidMethodInputError: Si moins de 3 variables sont fournies ou si
            moins de 2 observations complètes.

    Note:
        Hypothèses : données appariées, au moins 3 conditions.
        Référence Python : scipy.stats.friedmanchisquare
        Référence R : stats::friedman.test
    """
    if len(variables) < 3:
        raise InvalidMethodInputError(
            f"Au moins 3 variables sont nécessaires pour le test de Friedman. "
            f"Trouvé : {len(variables)}."
        )

    for var in variables:
        if var not in df.columns:
            raise InvalidMethodInputError(
                f"La colonne '{var}' doit exister dans le DataFrame."
            )

    df_clean, n_dropped = _drop_missing_pairs(df, variables)

    if len(df_clean) < 2:
        raise InvalidMethodInputError(
            f"Au moins 2 observations complètes sont nécessaires. "
            f"Seulement {len(df_clean)} disponible(s) après suppression."
        )

    # Extraction des colonnes
    arrays = [df_clean[var].values for var in variables]

    # Test de Friedman
    statistic, p_value = stats.friedmanchisquare(*arrays)

    # Taille d'effet : W de Kendall (coefficient de concordance)
    k = len(variables)
    n = len(df_clean)
    kendall_w = statistic / (n * (k - 1))

    # Médianes par variable
    medians = {var: float(np.median(arr)) for var, arr in zip(variables, arrays)}

    # Diagnostics
    diagnostics: list[DiagnosticFinding] = []
    diagnostics.extend(check_group_sizes(arrays, min_size=10))
    diagnostics.extend(_check_tied_ranks(arrays))

    # Code reproductible
    var_list = ", ".join([f'"{v}"' for v in variables])
    generated_code = f"""# Test de Friedman
import pandas as pd
from scipy import stats

df = pd.read_csv("votre_fichier.csv")
df_clean = df.dropna(subset=[{var_list}])
arrays = [df_clean[var].values for var in [{var_list}]]
statistic, p_value = stats.friedmanchisquare(*arrays)
print(f"Chi² = {{statistic:.4f}}, p = {{p_value:.4f}}")
"""

    return AnalysisResult(
        method="friedman",
        dataset_id="",
        variables={"variables": variables},
        sample_size=int(len(df_clean)),
        preprocessing=[
            f"Suppression des observations incomplètes : {n_dropped} ligne(s) retirée(s)"
        ],
        estimates={
            "statistic": float(statistic),
        },
        standard_errors={},
        confidence_intervals={},
        p_values={"overall": float(p_value)},
        model_metrics={
            "method_family": "inference",
            "effect_size": {"kendall_w": float(kendall_w)},
            "n_conditions": int(k),
            "condition_medians": medians,
            "n_obs_used": int(len(df_clean)),
            "n_obs_dropped": int(n_dropped),
        },
        diagnostics=diagnostics,
        warnings=[],
        assumptions_checked=[
            "Données appariées",
            "Au moins 3 conditions",
        ],
        generated_code=generated_code,
        execution_log=["Test de Friedman exécuté avec succès"],
        software_versions=_get_software_versions(),
    )


# ============================================================================
# Test de Spearman
# ============================================================================


def run_spearman_test(
    df: pd.DataFrame,
    var1: str,
    var2: str,
) -> AnalysisResult:
    """Test de corrélation de Spearman entre deux variables.

    Args:
        df: DataFrame contenant les données.
        var1: Nom de la première variable numérique.
        var2: Nom de la deuxième variable numérique.

    Returns:
        AnalysisResult avec les résultats du test.

    Raises:
        InvalidMethodInputError: Si les colonnes n'existent pas ou si moins
            de 3 observations valides.

    Note:
        Mesure la corrélation monotone entre deux variables.
        Référence Python : scipy.stats.spearmanr
        Référence R : stats::cor.test(..., method="spearman")
    """
    if var1 not in df.columns or var2 not in df.columns:
        raise InvalidMethodInputError(
            f"Les colonnes '{var1}' et '{var2}' doivent exister dans le DataFrame."
        )

    df_clean, n_dropped = _drop_missing_pairs(df, [var1, var2])

    if len(df_clean) < 3:
        raise InvalidMethodInputError(
            f"Au moins 3 observations valides sont nécessaires. "
            f"Seulement {len(df_clean)} disponible(s) après suppression."
        )

    x = df_clean[var1].values
    y = df_clean[var2].values

    # Test de Spearman
    statistic, p_value = stats.spearmanr(x, y)

    # Diagnostics
    diagnostics: list[DiagnosticFinding] = []
    diagnostics.extend(check_group_sizes([x, y], min_size=10))
    diagnostics.extend(_check_tied_ranks([x, y]))

    # Code reproductible
    generated_code = f"""# Test de corrélation de Spearman
import pandas as pd
from scipy import stats

df = pd.read_csv("votre_fichier.csv")
df_clean = df.dropna(subset=["{var1}", "{var2}"])
statistic, p_value = stats.spearmanr(df_clean["{var1}"].values, df_clean["{var2}"].values)
print(f"rho = {{statistic:.4f}}, p = {{p_value:.4f}}")
"""

    return AnalysisResult(
        method="spearman_test",
        dataset_id="",
        variables={"var1": var1, "var2": var2},
        sample_size=int(len(df_clean)),
        preprocessing=[
            f"Suppression des valeurs manquantes : {n_dropped} ligne(s) retirée(s)"
        ],
        estimates={
            "statistic": float(statistic),
            "rho": float(statistic),
        },
        standard_errors={},
        confidence_intervals={},
        p_values={"overall": float(p_value)},
        model_metrics={
            "method_family": "inference",
            "effect_size": {"rho": float(statistic)},
            "n_obs_used": int(len(df_clean)),
            "n_obs_dropped": int(n_dropped),
        },
        diagnostics=diagnostics,
        warnings=[],
        assumptions_checked=[
            "Relation monotone entre les variables",
        ],
        generated_code=generated_code,
        execution_log=["Test de corrélation de Spearman exécuté avec succès"],
        software_versions=_get_software_versions(),
    )


# ============================================================================
# Test de Kendall
# ============================================================================


def run_kendall_test(
    df: pd.DataFrame,
    var1: str,
    var2: str,
) -> AnalysisResult:
    """Test de corrélation de Kendall tau entre deux variables.

    Args:
        df: DataFrame contenant les données.
        var1: Nom de la première variable numérique.
        var2: Nom de la deuxième variable numérique.

    Returns:
        AnalysisResult avec les résultats du test.

    Raises:
        InvalidMethodInputError: Si les colonnes n'existent pas ou si moins
            de 3 observations valides.

    Note:
        Mesure la corrélation de rangs entre deux variables (tau-b).
        Référence Python : scipy.stats.kendalltau
        Référence R : stats::cor.test(..., method="kendall")
    """
    if var1 not in df.columns or var2 not in df.columns:
        raise InvalidMethodInputError(
            f"Les colonnes '{var1}' et '{var2}' doivent exister dans le DataFrame."
        )

    df_clean, n_dropped = _drop_missing_pairs(df, [var1, var2])

    if len(df_clean) < 3:
        raise InvalidMethodInputError(
            f"Au moins 3 observations valides sont nécessaires. "
            f"Seulement {len(df_clean)} disponible(s) après suppression."
        )

    x = df_clean[var1].values
    y = df_clean[var2].values

    # Test de Kendall tau-b
    statistic, p_value = stats.kendalltau(x, y)

    # Diagnostics
    diagnostics: list[DiagnosticFinding] = []
    diagnostics.extend(check_group_sizes([x, y], min_size=10))
    diagnostics.extend(_check_tied_ranks([x, y]))

    # Code reproductible
    generated_code = f"""# Test de corrélation de Kendall
import pandas as pd
from scipy import stats

df = pd.read_csv("votre_fichier.csv")
df_clean = df.dropna(subset=["{var1}", "{var2}"])
statistic, p_value = stats.kendalltau(df_clean["{var1}"].values, df_clean["{var2}"].values)
print(f"tau = {{statistic:.4f}}, p = {{p_value:.4f}}")
"""

    return AnalysisResult(
        method="kendall_test",
        dataset_id="",
        variables={"var1": var1, "var2": var2},
        sample_size=int(len(df_clean)),
        preprocessing=[
            f"Suppression des valeurs manquantes : {n_dropped} ligne(s) retirée(s)"
        ],
        estimates={
            "statistic": float(statistic),
            "tau": float(statistic),
        },
        standard_errors={},
        confidence_intervals={},
        p_values={"overall": float(p_value)},
        model_metrics={
            "method_family": "inference",
            "effect_size": {"tau": float(statistic)},
            "n_obs_used": int(len(df_clean)),
            "n_obs_dropped": int(n_dropped),
        },
        diagnostics=diagnostics,
        warnings=[],
        assumptions_checked=[
            "Relation monotone entre les variables",
        ],
        generated_code=generated_code,
        execution_log=["Test de corrélation de Kendall exécuté avec succès"],
        software_versions=_get_software_versions(),
    )


# ============================================================================
# Enregistrement dans le registre
# ============================================================================


_MANN_WHITNEY_SPEC = MethodSpec(
    method_id="mann_whitney",
    label_fr="Test de Mann-Whitney U",
    family="inference",
    engine="python",
    outcome_types=["continuous"],
    predictor_types=["binary"],
    min_sample_size=4,
    assumptions=[
        "Indépendance des observations",
        "Distributions de même forme",
    ],
    reference_impl="scipy.stats.mannwhitneyu",
    r_reference="stats::wilcox.test",
    phase=17,
)

_WILCOXON_SIGNED_RANK_SPEC = MethodSpec(
    method_id="wilcoxon_signed_rank",
    label_fr="Test de Wilcoxon signed-rank",
    family="inference",
    engine="python",
    outcome_types=["continuous"],
    predictor_types=["continuous"],
    min_sample_size=2,
    assumptions=[
        "Différences symétriquement distribuées",
        "Données appariées",
    ],
    reference_impl="scipy.stats.wilcoxon",
    r_reference="stats::wilcox.test(..., paired=TRUE)",
    phase=17,
)

_KRUSKAL_WALLIS_SPEC = MethodSpec(
    method_id="kruskal_wallis",
    label_fr="Test de Kruskal-Wallis",
    family="inference",
    engine="python",
    outcome_types=["continuous"],
    predictor_types=["categorical"],
    min_sample_size=4,
    assumptions=[
        "Indépendance des observations",
        "Distributions de même forme",
    ],
    reference_impl="scipy.stats.kruskal",
    r_reference="stats::kruskal.test",
    phase=17,
)

_FRIEDMAN_SPEC = MethodSpec(
    method_id="friedman",
    label_fr="Test de Friedman",
    family="inference",
    engine="python",
    outcome_types=["continuous"],
    predictor_types=[],
    min_sample_size=2,
    assumptions=[
        "Données appariées",
        "Au moins 3 conditions",
    ],
    reference_impl="scipy.stats.friedmanchisquare",
    r_reference="stats::friedman.test",
    phase=17,
)

_SPEARMAN_TEST_SPEC = MethodSpec(
    method_id="spearman_test",
    label_fr="Test de corrélation de Spearman",
    family="inference",
    engine="python",
    outcome_types=["continuous"],
    predictor_types=["continuous"],
    min_sample_size=3,
    assumptions=[
        "Relation monotone entre les variables",
    ],
    reference_impl="scipy.stats.spearmanr",
    r_reference="stats::cor.test(..., method=\"spearman\")",
    phase=17,
)

_KENDALL_TEST_SPEC = MethodSpec(
    method_id="kendall_test",
    label_fr="Test de corrélation de Kendall",
    family="inference",
    engine="python",
    outcome_types=["continuous"],
    predictor_types=["continuous"],
    min_sample_size=3,
    assumptions=[
        "Relation monotone entre les variables",
    ],
    reference_impl="scipy.stats.kendalltau",
    r_reference="stats::cor.test(..., method=\"kendall\")",
    phase=17,
)

# Enregistrement des méthodes
register_method(_MANN_WHITNEY_SPEC, run_mann_whitney)
register_method(_WILCOXON_SIGNED_RANK_SPEC, run_wilcoxon_signed_rank)
register_method(_KRUSKAL_WALLIS_SPEC, run_kruskal_wallis)
register_method(_FRIEDMAN_SPEC, run_friedman)
register_method(_SPEARMAN_TEST_SPEC, run_spearman_test)
register_method(_KENDALL_TEST_SPEC, run_kendall_test)
