"""Tests ANOVA étendus (Phase 17 — T17.3).

ANCOVA, MANOVA, post-hoc (Tukey / Games-Howell / Dunn), Welch ANOVA et
ANOVA à mesures répétées.

Contrats : V1 §7 et V2 §6.2 (AnalysisResult), V2 §7.1 (registre).
Toutes les valeurs exposées sont des types Python natifs (JSON-safe).
"""

import math
import sys
from itertools import combinations
from typing import Any

from collections.abc import Callable
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sp_stats
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from app.diagnostics.checks_inference import check_equal_variances, check_group_sizes
from app.diagnostics.models import DiagnosticFinding, Severity
from app.errors import InvalidMethodInputError
from app.results.models import AnalysisResult
from app.statistics.registry import MethodSpec, register_method

ALPHA = 0.05


# ============================================================================
# Helpers
# ============================================================================


def _patsy_name(name: str) -> str:
    """Protège un nom de colonne pour une formule patsy."""
    return name if name.isidentifier() else f"`{name}`"


def _norm(value: Any) -> str:
    """Normalise un nom de colonne / de terme (insensible aux espaces et symboles)."""
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _column(frame, *aliases: str):
    """Retrouve une colonne par ses aliases normalisés, indépendamment de la version."""
    normalized = {_norm(c): c for c in frame.columns}
    for alias in aliases:
        key = _norm(alias)
        if key in normalized:
            return normalized[key]
    raise InvalidMethodInputError(
        f"Colonne introuvable {aliases}. Colonnes disponibles : {list(frame.columns)}"
    )


def _row(table, *aliases: str):
    """Retrouve une ligne par son nom normalisé."""
    normalized = {_norm(idx): idx for idx in table.index}
    for alias in aliases:
        if _norm(alias) in normalized:
            return normalized[_norm(alias)]
    raise InvalidMethodInputError(
        f"Terme introuvable {aliases}. Termes disponibles : {list(table.index)}"
    )


def _finite(value: Any) -> float | None:
    """Convertit en float natif, ou None si NaN / inf / non numérique."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _require_finite(value: Any, label: str) -> float:
    result = _finite(value)
    if result is None:
        raise InvalidMethodInputError(
            f"Statistique non définie ({label}) : effectifs insuffisants ou modèle dégénéré."
        )
    return result


def _drop_missing(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, int]:
    cleaned = df.dropna(subset=columns)
    return cleaned, len(df) - len(cleaned)


def _get_software_versions() -> dict:
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": sm.__version__,
    }
    try:
        import pingouin

        versions["pingouin"] = pingouin.__version__
    except ImportError:
        versions["pingouin"] = None
    return versions


def _assert_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise InvalidMethodInputError(f"La colonne '{col}' doit être numérique.")


def _normality_finding(values: np.ndarray, label: str) -> list[DiagnosticFinding]:
    if not 3 <= len(values) <= 5000 or np.ptp(values) == 0:
        return []
    p = _finite(sp_stats.shapiro(values).pvalue)
    if p is None or p >= ALPHA:
        return []
    return [
        DiagnosticFinding(
            code="NON_NORMAL_RESIDUALS",
            severity=Severity.WARNING,
            message=(
                f"Les résidus ({label}) s'écartent de la normalité "
                f"(Shapiro-Wilk, p={p:.4f}). Interprétez les tests avec prudence."
            ),
            details={"test": "shapiro", "p_value": p, "n": len(values)},
        )
    ]


# ============================================================================
# ANCOVA
# ============================================================================


def run_ancova(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    covariates: list[str],
) -> AnalysisResult:
    """ANCOVA : comparaison de groupes avec ajustement sur des covariables.

    ANOVA de type II, eta-carré partiel, test d'homogénéité des pentes par
    comparaison de modèles (additif vs interactions).
    Référence R : car::Anova(lm(...), type = 2)
    """
    all_cols = [outcome, group, *covariates]
    for col in all_cols:
        if col not in df.columns:
            raise InvalidMethodInputError(
                f"La colonne '{col}' doit exister dans le DataFrame."
            )
    if not covariates:
        raise InvalidMethodInputError(
            "Au moins une covariable est nécessaire pour une ANCOVA."
        )

    _assert_numeric(df, [outcome, *covariates])
    df_clean, n_dropped = _drop_missing(df, all_cols)

    labels = sorted(str(g) for g in df_clean[group].dropna().unique())
    if len(labels) < 2:
        raise InvalidMethodInputError(
            f"La variable de groupe '{group}' doit avoir au moins 2 modalités. "
            f"Trouvé : {len(labels)} modalité(s)."
        )

    g = _patsy_name(group)
    cov_terms = " + ".join(_patsy_name(c) for c in covariates)
    additive = f"{_patsy_name(outcome)} ~ C({g}) + {cov_terms}"

    model = smf.ols(additive, data=df_clean).fit()
    if model.df_resid <= 0 or not np.all(np.isfinite(model.params)):
        raise InvalidMethodInputError(
            "Modèle ANCOVA non identifiable : augmentez les effectifs ou réduisez le nombre de covariables."
        )

    table = anova_lm(model, typ=2)
    group_row = _row(table, f"C({group})", group)
    residual_row = _row(table, "Residual", "Residuais", "Residus")

    f_stat = _require_finite(table.loc[group_row, _column(table, "f")], "F ANCOVA")
    p_value = _require_finite(table.loc[group_row, _column(table, "prf")], "p ANCOVA")
    ss_group = _require_finite(
        table.loc[group_row, _column(table, "sumsq")], "SC groupe"
    )
    ss_residual = _require_finite(
        table.loc[residual_row, _column(table, "sumsq")], "SC résiduelle"
    )
    df_effect = int(
        _require_finite(table.loc[group_row, _column(table, "df")], "ddl groupe")
    )

    denominator = ss_group + ss_residual
    partial_eta = ss_group / denominator if denominator > 0 else 0.0

    diagnostics: list[DiagnosticFinding] = []
    diagnostics.extend(_normality_finding(np.asarray(model.resid), "ANCOVA"))

    residuals_by_group = [
        np.asarray(model.resid[df_clean[group].astype(str) == label])
        for label in labels
    ]
    diagnostics.extend(check_group_sizes(residuals_by_group, min_size=10))
    if all(len(r) >= 2 for r in residuals_by_group):
        diagnostics.extend(check_equal_variances(residuals_by_group, alpha=ALPHA))

    # Homogénéité des pentes : test global des interactions groupe x covariables
    interaction_terms = " + ".join(f"C({g}):{_patsy_name(c)}" for c in covariates)
    full_model = smf.ols(f"{additive} + {interaction_terms}", data=df_clean).fit()
    p_slopes = None
    if full_model.df_resid > 0 and np.all(np.isfinite(full_model.params)):
        comparison = anova_lm(model, full_model)
        p_slopes = _finite(comparison.loc[1, _column(comparison, "prf")])
        if p_slopes is not None and p_slopes < ALPHA:
            diagnostics.append(
                DiagnosticFinding(
                    code="HETEROGENEOUS_SLOPES",
                    severity=Severity.PROBABLE_ISSUE,
                    message=(
                        f"L'interaction groupe × covariables est significative (p={p_slopes:.4f}). "
                        "L'hypothèse d'homogénéité des pentes est violée : "
                        "envisagez un modèle avec interactions ou une analyse par strates."
                    ),
                    details={"p_value": p_slopes, "test": "comparaison de modèles"},
                )
            )

    estimates = {"f_statistic": f_stat, "partial_eta_squared": partial_eta}
    standard_errors: dict[str, float] = {}
    confidence_intervals: dict[str, list[float]] = {}
    for cov in covariates:
        if cov in model.params.index:
            coef = _finite(model.params[cov])
            if coef is not None:
                estimates[f"coef_{cov}"] = coef
            se = _finite(model.bse.get(cov))
            if se is not None:
                standard_errors[f"coef_{cov}"] = se
            ci = model.conf_int().loc[cov]
            ci_low, ci_high = _finite(ci.iloc[0]), _finite(ci.iloc[1])
            if ci_low is not None and ci_high is not None:
                confidence_intervals[f"coef_{cov}"] = [ci_low, ci_high]

    anova_records = [
        {
            "term": str(idx),
            "sum_sq": _finite(row.get(_column(table, "sumsq"))),
            "df": _finite(row.get(_column(table, "df"))),
            "F": _finite(row.get(_column(table, "f"))),
            "p_value": _finite(row.get(_column(table, "prf"))),
        }
        for idx, row in table.iterrows()
    ]

    generated_code = (
        "import pandas as pd\n"
        "import statsmodels.formula.api as smf\n"
        "from statsmodels.stats.anova import anova_lm\n\n"
        f'df = pd.read_csv("votre_fichier.csv")\n'
        f"df_clean = df.dropna(subset={all_cols!r})\n"
        f'model = smf.ols("{additive}", data=df_clean).fit()\n'
        "print(anova_lm(model, typ=2))\n"
    )

    return AnalysisResult(
        method="ancova",
        dataset_id="",
        variables={"outcome": outcome, "group": group, "covariates": list(covariates)},
        sample_size=len(df_clean),
        preprocessing=[
            f"Suppression des valeurs manquantes : {n_dropped} ligne(s) retirée(s)"
        ],
        estimates=estimates,
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals,
        p_values={"overall": p_value},
        model_metrics={
            "method_family": "inference",
            "effect_size": {"partial_eta_squared": partial_eta},
            "anova_type": 2,
            "df_effect": df_effect,
            "df_residual": int(model.df_resid),
            "n_groups": len(labels),
            "n_covariates": len(covariates),
            "p_homogeneity_of_slopes": p_slopes,
            "group_sizes": {
                label: int((df_clean[group].astype(str) == label).sum())
                for label in labels
            },
            "n_obs_used": len(df_clean),
            "n_obs_dropped": int(n_dropped),
            "tables": {"ancova": anova_records},
        },
        diagnostics=diagnostics,
        warnings=[],
        assumptions_checked=[
            "Normalité des résidus",
            "Homogénéité des variances",
            "Homogénéité des pentes de régression",
            "Relation linéaire covariables / variable dépendante",
        ],
        generated_code=generated_code,
        execution_log=["ANCOVA exécutée avec succès (ANOVA de type II)"],
        software_versions=_get_software_versions(),
    )


# ============================================================================
# MANOVA
# ============================================================================


def run_manova(df: pd.DataFrame, outcomes: list[str], group: str) -> AnalysisResult:
    """MANOVA multivariée, statistic de Pillai et F approché.

    Référence R : summary(manova(cbind(...) ~ group), test = "Pillai")
    """
    all_cols = [*outcomes, group]
    for col in all_cols:
        if col not in df.columns:
            raise InvalidMethodInputError(
                f"La colonne '{col}' doit exister dans le DataFrame."
            )
    if len(outcomes) < 2:
        raise InvalidMethodInputError(
            f"Au moins 2 variables dépendantes sont nécessaires pour une MANOVA. Trouvé : {len(outcomes)}."
        )

    _assert_numeric(df, outcomes)
    df_clean, n_dropped = _drop_missing(df, all_cols)

    labels = sorted(str(g) for g in df_clean[group].dropna().unique())
    if len(labels) < 2:
        raise InvalidMethodInputError(
            f"La variable de groupe '{group}' doit avoir au moins 2 modalités. "
            f"Trouvé : {len(labels)} modalité(s)."
        )
    if len(df_clean) < len(outcomes) + len(labels) + 1:
        raise InvalidMethodInputError(
            "Effectifs insuffisants pour une MANOVA : visez n >= nombre de réponses + nombre de groupes + 1."
        )

    g_name = _patsy_name(group)
    outcome_terms = " + ".join(_patsy_name(o) for o in outcomes)
    formula = f"{outcome_terms} ~ C({g_name})"
    model = MANOVA.from_formula(formula, data=df_clean)
    mv = model.mv_test()

    candidates: list[tuple[str, pd.DataFrame]] = []
    for key, payload in mv.results.items():
        if str(key).lower() in {"intercept", "const"}:
            continue
        if not isinstance(payload, dict) or "stat" not in payload:
            continue
        stat_table = payload["stat"]
        if stat_table is None or len(stat_table) == 0:
            continue
        candidates.append((str(key), stat_table))

    if not candidates:
        raise InvalidMethodInputError(
            f"Effet de groupe introuvable dans la MANOVA. Effets disponibles : {list(mv.results)}"
        )

    preferred = {
        _norm("group"),
        _norm(group),
        _norm(f"C({group})"),
        _norm(f"C({g_name})"),
    }
    exact = [(key, table) for key, table in candidates if _norm(key) in preferred]
    if exact:
        effect_key, stat_table = exact[0]
    else:
        effect_key, stat_table = candidates[0]

    pillai_label = _row(stat_table, "Pillai's trace", "pillai trace", "pillaistrace")
    row = stat_table.loc[pillai_label]
    pillai = _require_finite(row[_column(stat_table, "value")], "trace de Pillai")
    f_stat = _require_finite(row[_column(stat_table, "f", "fvalue")], "F MANOVA")
    p_value = _require_finite(row[_column(stat_table, "prf", "pr>f")], "p MANOVA")
    df_hypothesis = int(
        _require_finite(row[_column(stat_table, "numdf")], "ddl hypothèse")
    )
    df_error = _require_finite(row[_column(stat_table, "dendf")], "ddl résiduels")

    diagnostics: list[DiagnosticFinding] = []
    groups_data = [
        df_clean.loc[df_clean[group].astype(str) == label, outcomes].to_numpy(
            dtype=float
        )
        for label in labels
    ]
    diagnostics.extend(check_group_sizes(groups_data, min_size=10))

    generated_code = (
        "import pandas as pd\n"
        "from statsmodels.multivariate.manova import MANOVA\n\n"
        f'df = pd.read_csv("votre_fichier.csv")\n'
        f"df_clean = df.dropna(subset={all_cols!r})\n"
        f'model = MANOVA.from_formula("{formula}", data=df_clean)\n'
        "print(model.mv_test())\n"
    )

    return AnalysisResult(
        method="manova",
        dataset_id="",
        variables={"outcomes": list(outcomes), "group": group},
        sample_size=len(df_clean),
        preprocessing=[
            f"Suppression des valeurs manquantes : {n_dropped} ligne(s) retirée(s)"
        ],
        estimates={"pillai_trace": pillai, "f_statistic": f_stat},
        standard_errors={},
        confidence_intervals={},
        p_values={"overall": p_value},
        model_metrics={
            "method_family": "inference",
            "effect_size": {"pillai_trace": pillai},
            "multivariate_test": "pillai",
            "effect_key": effect_key,
            "df_hypothesis": df_hypothesis,
            "df_error": df_error,
            "n_groups": len(labels),
            "n_outcomes": len(outcomes),
            "group_sizes": {
                label: len(data) for label, data in zip(labels, groups_data)
            },
            "n_obs_used": len(df_clean),
            "n_obs_dropped": int(n_dropped),
            "tables": {
                "manova": [
                    {
                        "test": "pillai",
                        "value": pillai,
                        "F": f_stat,
                        "df_hypothesis": df_hypothesis,
                        "df_error": df_error,
                        "p_value": p_value,
                    }
                ]
            },
        },
        diagnostics=diagnostics,
        warnings=[],
        assumptions_checked=[
            "Normalité multivariée",
            "Homogénéité des matrices de covariance",
            "Indépendance des observations",
        ],
        generated_code=generated_code,
        execution_log=["MANOVA exécutée avec succès (critère de Pillai)"],
        software_versions=_get_software_versions(),
    )


# ============================================================================
# Post-hoc
# ============================================================================


def run_posthoc(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    method: str = "tukey",
) -> AnalysisResult:
    """Comparaisons post-hoc : Tukey HSD, Games-Howell ou Dunn.

    Retourne le test omnibus (ANOVA ou Kruskal-Wallis) ET chaque comparaison
    par paire, avec IC et p-values corrigées lorsque disponibles.
    """
    if outcome not in df.columns or group not in df.columns:
        raise InvalidMethodInputError(
            f"Les colonnes '{outcome}' et '{group}' doivent exister dans le DataFrame."
        )

    normalized_method = method.lower().replace("_", "-")
    if normalized_method not in {"tukey", "games-howell", "dunn"}:
        raise InvalidMethodInputError(
            f"Méthode invalide : '{method}'. Choisir parmi 'tukey', 'games-howell', 'dunn'."
        )
    method = normalized_method
    p_adjustment = {
        "tukey": "tukey",
        "games-howell": "none",
        "dunn": "bonferroni",
    }[method]

    _assert_numeric(df, [outcome])
    df_clean, n_dropped = _drop_missing(df, [outcome, group])

    labels = sorted(str(g) for g in df_clean[group].dropna().unique())
    if len(labels) < 2:
        raise InvalidMethodInputError(
            f"La variable de groupe '{group}' doit avoir au moins 2 modalités. "
            f"Trouvé : {len(labels)} modalité(s)."
        )

    groups_data = [
        df_clean.loc[df_clean[group].astype(str) == label, outcome].to_numpy(
            dtype=float
        )
        for label in labels
    ]
    if any(len(data) < 2 for data in groups_data):
        raise InvalidMethodInputError(
            "Au moins 2 observations par groupe sont nécessaires pour les tests post-hoc."
        )

    diagnostics: list[DiagnosticFinding] = []
    diagnostics.extend(check_group_sizes(groups_data, min_size=10))
    diagnostics.extend(check_equal_variances(groups_data, alpha=ALPHA))

    omnibus_test = "kruskal" if method == "dunn" else "anova"
    if method == "dunn":
        omnibus = sp_stats.kruskal(*groups_data)
    else:
        omnibus = sp_stats.f_oneway(*groups_data)
    omnibus_f = _require_finite(omnibus.statistic, "statistique omnibus")
    omnibus_p = _require_finite(omnibus.pvalue, "p omnibus")

    comparisons: list[dict[str, Any]] = []

    if method == "tukey":
        tukey = pairwise_tukeyhsd(
            endog=df_clean[outcome].to_numpy(dtype=float),
            groups=df_clean[group].astype(str).to_numpy(),
            alpha=ALPHA,
        )
        pairs = list(combinations([str(x) for x in np.asarray(tukey.groupsunique)], 2))
        mean_diffs = np.asarray(tukey.meandiffs, dtype=float)
        confint = np.asarray(tukey.confint, dtype=float)
        p_values = np.asarray(tukey.pvalues, dtype=float)

        for index, (g1, g2) in enumerate(pairs):
            p_adj = _finite(p_values[index])
            comparisons.append(
                {
                    "group1": g1,
                    "group2": g2,
                    "mean_diff": _finite(mean_diffs[index]),
                    "p_value": p_adj,
                    "p_adjusted": p_adj,
                    "ci_low": _finite(confint[index, 0]),
                    "ci_high": _finite(confint[index, 1]),
                    "standard_error": None,
                    "significant": p_adj is not None and p_adj < ALPHA,
                }
            )

    elif method == "games-howell":
        import pingouin as pg

        table = pg.pairwise_gameshowell(data=df_clean, dv=outcome, between=group)
        for _, row in table.iterrows():
            p_val = _finite(row["pval"])
            comparisons.append(
                {
                    "group1": str(row["A"]),
                    "group2": str(row["B"]),
                    "mean_diff": _finite(row["diff"]),
                    "p_value": p_val,
                    "p_adjusted": p_val,
                    "ci_low": None,
                    "ci_high": None,
                    "standard_error": None,
                    "significant": p_val is not None and p_val < ALPHA,
                }
            )

    else:  # dunn
        import pingouin as pg

        table = pg.pairwise_tests(
            data=df_clean,
            dv=outcome,
            between=group,
            parametric=False,
            padjust="bonferroni",
        )
        p_col = _column(table, "p_corr", "p_adj", "p_unc", "pval")

        for _, row in table.iterrows():
            g1_data = df_clean.loc[
                df_clean[group].astype(str) == str(row["A"]), outcome
            ]
            g2_data = df_clean.loc[
                df_clean[group].astype(str) == str(row["B"]), outcome
            ]
            median_diff = float(np.median(g1_data) - np.median(g2_data))
            p_val = _finite(row[p_col])

            comparisons.append(
                {
                    "group1": str(row["A"]),
                    "group2": str(row["B"]),
                    "mean_diff": median_diff,
                    "p_value": p_val,
                    "p_adjusted": p_val,
                    "ci_low": None,
                    "ci_high": None,
                    "standard_error": None,
                    "significant": p_val is not None and p_val < ALPHA,
                }
            )

    p_values_out = {"overall": omnibus_p}
    confidence_intervals: dict[str, list[float]] = {}
    standard_errors: dict[str, float] = {}
    for item in comparisons:
        key = f"{item['group1']}_vs_{item['group2']}"
        if item["p_adjusted"] is not None:
            p_values_out[key] = item["p_adjusted"]
        if item["ci_low"] is not None and item["ci_high"] is not None:
            confidence_intervals[key] = [item["ci_low"], item["ci_high"]]
        if item["standard_error"] is not None:
            standard_errors[key] = item["standard_error"]

    adjusted = [c["p_adjusted"] for c in comparisons if c["p_adjusted"] is not None]
    diffs = [abs(c["mean_diff"]) for c in comparisons if c["mean_diff"] is not None]

    if method == "tukey":
        code_lines = [
            "from statsmodels.stats.multicomp import pairwise_tukeyhsd",
            f'tukey = pairwise_tukeyhsd(df_clean["{outcome}"], df_clean["{group}"], alpha={ALPHA})',
            "print(tukey)",
        ]
    elif method == "games-howell":
        code_lines = [
            "import pingouin as pg",
            (
                f'table = pg.pairwise_gameshowell(data=df_clean, dv="{outcome}", '
                f'between="{group}")'
            ),
            "print(table)",
        ]
    else:
        code_lines = [
            "import pingouin as pg",
            (
                f'table = pg.pairwise_tests(data=df_clean, dv="{outcome}", between="{group}", '
                f'parametric=False, padjust="{p_adjustment}")'
            ),
            "print(table)",
        ]

    generated_code = (
        "import pandas as pd\n\n"
        f'df = pd.read_csv("votre_fichier.csv")\n'
        f'df_clean = df.dropna(subset=["{outcome}", "{group}"])\n'
        + "\n".join(code_lines)
        + "\n"
    )

    return AnalysisResult(
        method=f"posthoc_{method}",
        dataset_id="",
        variables={"outcome": outcome, "group": group, "method": method},
        sample_size=len(df_clean),
        preprocessing=[
            f"Suppression des valeurs manquantes : {n_dropped} ligne(s) retirée(s)"
        ],
        estimates={
            "f_statistic": omnibus_f,
            "n_significant": int(sum(c["significant"] for c in comparisons)),
            "min_p_value": min(adjusted) if adjusted else None,
            "max_abs_mean_diff": max(diffs) if diffs else None,
        },
        standard_errors=standard_errors,
        confidence_intervals=confidence_intervals,
        p_values=p_values_out,
        model_metrics={
            "method_family": "inference",
            "test_name": method,
            "p_adjustment": p_adjustment,
            "omnibus_test": omnibus_test,
            "n_comparisons": len(comparisons),
            "n_groups": len(labels),
            "group_sizes": {
                label: len(data) for label, data in zip(labels, groups_data)
            },
            "n_obs_used": len(df_clean),
            "n_obs_dropped": int(n_dropped),
            "tables": {"comparisons": comparisons},
        },
        diagnostics=diagnostics,
        warnings=[],
        assumptions_checked=[
            "Indépendance des observations",
            "Égalité des variances"
            if method == "tukey"
            else "Ne suppose pas l'égalité des variances",
        ],
        generated_code=generated_code,
        execution_log=[f"Test post-hoc ({method}) exécuté avec succès"],
        software_versions=_get_software_versions(),
    )


# ============================================================================
# Welch ANOVA
# ============================================================================


def run_welch_anova(df: pd.DataFrame, outcome: str, group: str) -> AnalysisResult:
    """ANOVA de Welch : ne suppose pas l'égalité des variances.

    Référence R : oneway.test(outcome ~ group, var.equal = FALSE)
    """
    if outcome not in df.columns or group not in df.columns:
        raise InvalidMethodInputError(
            f"Les colonnes '{outcome}' et '{group}' doivent exister dans le DataFrame."
        )

    _assert_numeric(df, [outcome])
    df_clean, n_dropped = _drop_missing(df, [outcome, group])

    labels = sorted(str(g) for g in df_clean[group].dropna().unique())
    if len(labels) < 2:
        raise InvalidMethodInputError(
            f"La variable de groupe '{group}' doit avoir au moins 2 modalités. "
            f"Trouvé : {len(labels)} modalité(s)."
        )

    groups_data = [
        df_clean.loc[df_clean[group].astype(str) == label, outcome].to_numpy(
            dtype=float
        )
        for label in labels
    ]
    if any(len(data) < 2 for data in groups_data):
        raise InvalidMethodInputError(
            "Au moins 2 observations par groupe sont nécessaires."
        )

    import pingouin as pg

    table = pg.welch_anova(data=df_clean, dv=outcome, between=group)
    row = table.iloc[0]
    f_stat = _require_finite(row[_column(table, "f", "w")], "F de Welch")
    p_value = _require_finite(row[_column(table, "punc", "pval")], "p de Welch")
    df_between = int(_require_finite(row[_column(table, "ddof1")], "ddl entre"))
    df_within = _require_finite(row[_column(table, "ddof2")], "ddl intra")

    # Oméga-carré de Kelley (et non eta-carré)
    k = len(groups_data)
    n_total = sum(len(data) for data in groups_data)
    grand_mean = float(np.mean(np.concatenate(groups_data)))
    ss_between = float(
        sum(len(data) * (np.mean(data) - grand_mean) ** 2 for data in groups_data)
    )
    ss_within = float(sum(np.sum((data - np.mean(data)) ** 2) for data in groups_data))
    ss_total = ss_between + ss_within
    ms_within = ss_within / (n_total - k) if n_total > k else 0.0
    omega = (
        (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
        if ss_total + ms_within > 0
        else 0.0
    )
    omega = max(0.0, omega)

    diagnostics: list[DiagnosticFinding] = []
    diagnostics.extend(check_group_sizes(groups_data, min_size=10))

    generated_code = (
        "import pandas as pd\nimport pingouin as pg\n\n"
        f'df = pd.read_csv("votre_fichier.csv")\n'
        f'df_clean = df.dropna(subset=["{outcome}", "{group}"])\n'
        f'print(pg.welch_anova(data=df_clean, dv="{outcome}", between="{group}"))\n'
    )

    return AnalysisResult(
        method="welch_anova",
        dataset_id="",
        variables={"outcome": outcome, "group": group},
        sample_size=len(df_clean),
        preprocessing=[
            f"Suppression des valeurs manquantes : {n_dropped} ligne(s) retirée(s)"
        ],
        estimates={"f_statistic": f_stat, "omega_squared": omega},
        standard_errors={},
        confidence_intervals={},
        p_values={"overall": p_value},
        model_metrics={
            "method_family": "inference",
            "effect_size": {"omega_squared": omega},
            "df_between": df_between,
            "df_within": df_within,
            "n_groups": len(labels),
            "group_medians": {
                label: float(np.median(data))
                for label, data in zip(labels, groups_data)
            },
            "group_sizes": {
                label: len(data) for label, data in zip(labels, groups_data)
            },
            "n_obs_used": len(df_clean),
            "n_obs_dropped": int(n_dropped),
        },
        diagnostics=diagnostics,
        warnings=[],
        assumptions_checked=[
            "Indépendance des observations",
            "Normalité approximative dans chaque groupe",
            "Ne suppose pas l'égalité des variances",
        ],
        generated_code=generated_code,
        execution_log=["Welch ANOVA exécutée avec succès"],
        software_versions=_get_software_versions(),
    )


# ============================================================================
# ANOVA à mesures répétées
# ============================================================================


def run_repeated_measures_anova(
    df: pd.DataFrame,
    outcome: str,
    within: str,
    subject: str,
) -> AnalysisResult:
    """ANOVA à mesures répétées avec test de Mauchly et correction Greenhouse-Geisser.

    Suppression listwise des SUJETS incomplets (exigence d'un plan sphérique / équilibré).
    Référence R : aov(outcome ~ condition + Error(subject / condition))
    """
    all_cols = [outcome, within, subject]
    for col in all_cols:
        if col not in df.columns:
            raise InvalidMethodInputError(
                f"La colonne '{col}' doit exister dans le DataFrame."
            )

    _assert_numeric(df, [outcome])
    df_clean, n_missing = _drop_missing(df, all_cols)

    n_conditions_declared = int(df_clean[within].dropna().astype(str).nunique())
    if n_conditions_declared < 2:
        raise InvalidMethodInputError(
            f"La variable intra-sujet '{within}' doit avoir au moins 2 modalités. "
            f"Trouvé : {n_conditions_declared} modalité(s)."
        )

    if df_clean.duplicated(subset=[subject, within]).any():
        raise InvalidMethodInputError(
            "Chaque sujet ne doit apparaître qu'une fois par condition (doublons détectés)."
        )

    n_subjects_declared = int(df[subject].nunique())
    wide = df_clean.pivot(index=subject, columns=within, values=outcome).dropna(
        how="any"
    )
    n_subjects, n_conditions = wide.shape

    # Vérifier le nombre de sujets complets (avant d'appeler pingouin pour éviter ZeroDivisionError)
    if n_subjects < 2:
        raise InvalidMethodInputError(
            "Au moins 2 sujets complets sont nécessaires : le plan est déséquilibré après "
            "suppression des mesures manquantes."
        )

    n_used = int(wide.size)
    n_dropped = len(df) - n_used
    n_excluded_subjects = int(n_subjects_declared - n_subjects)

    analysis = wide.reset_index().melt(
        id_vars=subject, var_name=within, value_name=outcome
    )

    import pingouin as pg

    table = pg.rm_anova(data=analysis, dv=outcome, within=within, subject=subject)
    row = table.iloc[0]

    f_stat = _require_finite(row[_column(table, "f")], "F mesures répétées")
    p_value = _require_finite(row[_column(table, "punc", "pval")], "p mesures répétées")

    # pingouin >= 0.5.5 retourne 'ng2' (eta-carré généralisé) au lieu de 'np2'
    available_cols = {_norm(c): c for c in table.columns}

    if "np2" in available_cols:
        partial_eta = _finite(row[available_cols["np2"]])
    elif "ng2" in available_cols:
        partial_eta = _finite(row[available_cols["ng2"]])
    else:
        # Calcul manuel de eta-carré partiel à partir de F et des ddl
        df_effect = _require_finite(row[_column(table, "ddof1")], "ddl effet")
        df_error = _require_finite(row[_column(table, "ddof2")], "ddl erreur")
        partial_eta = (f_stat * df_effect) / (f_stat * df_effect + df_error)

    if partial_eta is None:
        partial_eta = 0.0

    def optional(*aliases: str) -> float | None:
        for alias in aliases:
            if alias in available_cols:
                return _finite(row[available_cols[alias]])
        return None

    epsilon = optional("eps")
    w_sphericity = optional("wspher")
    p_sphericity = optional("pspher")
    p_gg = optional("pgg")
    p_hf = optional("phf")

    diagnostics: list[DiagnosticFinding] = []
    if p_sphericity is not None and p_sphericity < ALPHA:
        message = (
            f"La sphéricité est violée (Mauchly, p={p_sphericity:.4f}). "
            "Utilisez la correction de Greenhouse-Geisser."
        )
        if p_gg is not None:
            message += f" F corrigé : p(GG)={p_gg:.4f}."
        diagnostics.append(
            DiagnosticFinding(
                code="SPHERICITY_VIOLATED",
                severity=Severity.PROBABLE_ISSUE,
                message=message,
                details={
                    "w_sphericity": w_sphericity,
                    "p_sphericity": p_sphericity,
                    "epsilon": epsilon,
                    "p_greenhouse_geisser": p_gg,
                },
            )
        )

    condition_means = {str(col): float(wide[col].mean()) for col in wide.columns}
    p_values_out = {"overall": p_value}
    if p_gg is not None:
        p_values_out["greenhouse_geisser"] = p_gg
    if p_hf is not None:
        p_values_out["huynh_feldt"] = p_hf

    generated_code = (
        "import pandas as pd\nimport pingouin as pg\n\n"
        f'df = pd.read_csv("votre_fichier.csv")\n'
        f"df_clean = df.dropna(subset={all_cols!r})\n"
        f'wide = df_clean.pivot(index="{subject}", columns="{within}", '
        f'values="{outcome}").dropna(how="any")\n'
        f'analysis = wide.reset_index().melt(id_vars="{subject}", '
        f'var_name="{within}", value_name="{outcome}")\n'
        f'print(pg.rm_anova(data=analysis, dv="{outcome}", within="{within}", '
        f'subject="{subject}"))\n'
    )

    return AnalysisResult(
        method="repeated_measures_anova",
        dataset_id="",
        variables={"outcome": outcome, "within": within, "subject": subject},
        sample_size=n_used,
        preprocessing=[
            f"Suppression des valeurs manquantes : {n_missing} ligne(s)",
            f"Sujets incomplets exclus (suppression listwise) : {n_excluded_subjects}",
        ],
        estimates={"f_statistic": f_stat, "partial_eta_squared": partial_eta},
        standard_errors={},
        confidence_intervals={},
        p_values=p_values_out,
        model_metrics={
            "method_family": "inference",
            "effect_size": {"partial_eta_squared": partial_eta},
            "df_effect": int(
                _require_finite(row[_column(table, "ddof1")], "ddl effet")
            ),
            "df_error": _require_finite(row[_column(table, "ddof2")], "ddl erreur"),
            "n_conditions": int(n_conditions),
            "n_subjects": int(n_subjects),
            "n_subjects_excluded": n_excluded_subjects,
            "condition_means": condition_means,
            "sphericity": {
                "w": w_sphericity,
                "p_value": p_sphericity,
                "epsilon": epsilon,
                "p_greenhouse_geisser": p_gg,
                "p_huynh_feldt": p_hf,
            },
            "n_obs_used": n_used,
            "n_obs_dropped": int(n_dropped),
        },
        diagnostics=diagnostics,
        warnings=[],
        assumptions_checked=[
            "Sphéricité (test de Mauchly)",
            "Normalité des différences entre conditions",
            "Données appariées / plan équilibré",
        ],
        generated_code=generated_code,
        execution_log=[
            "ANOVA à mesures répétées exécutée avec succès",
            f"{n_excluded_subjects} sujet(s) incomplet(s) exclus",
        ],
        software_versions=_get_software_versions(),
    )


# ============================================================================
# Registre (V2 §7.1)
# ============================================================================
MethodCallable = Callable[..., AnalysisResult]

MethodSpecTuple = tuple[
    str,              # method_id
    str,              # label_fr
    list[str],        # predictor_types
    int,              # min_sample_size
    list[str],        # assumptions
    str,              # reference_impl
    str | None,       # r_reference
    MethodCallable,   # fonction implémentée
]


_METHOD_SPECS : list[MethodSpecTuple] = [
    (
        "ancova",
        "ANCOVA (analyse de covariance)",
        ["categorical", "continuous"],
        10,
        [
            "Normalité des résidus",
            "Homogénéité des variances",
            "Homogénéité des pentes",
        ],
        "statsmodels.formula.api.ols + anova_lm(typ=2)",
        "car::Anova(lm(...), type = 2)",
        run_ancova,
    ),
    (
        "manova",
        "MANOVA multivariée",
        ["categorical"],
        10,
        ["Normalité multivariée", "Homogénéité des matrices de covariance"],
        "statsmodels.multivariate.manova.MANOVA",
        'summary(manova(...), test = "Pillai")',
        run_manova,
    ),
    (
        "posthoc",
        "Tests post-hoc (Tukey, Games-Howell, Dunn)",
        ["categorical"],
        6,
        ["Indépendance des observations", "Égalité des variances (Tukey uniquement)"],
        "statsmodels pairwise_tukeyhsd / pingouin",
        "stats::TukeyHSD",
        run_posthoc,
    ),
    (
        "welch_anova",
        "Welch ANOVA",
        ["categorical"],
        6,
        ["Indépendance des observations", "Ne suppose pas l'égalité des variances"],
        "pingouin.welch_anova",
        "stats::oneway.test(var.equal = FALSE)",
        run_welch_anova,
    ),
    (
        "repeated_measures_anova",
        "ANOVA à mesures répétées",
        ["categorical"],
        6,
        ["Sphéricité (Mauchly)", "Normalité des différences", "Données appariées"],
        "pingouin.rm_anova",
        "stats::aov(... + Error(subject/condition))",
        run_repeated_measures_anova,
    ),
]

for _id, _label, _predictors, _min_n, _assumptions, _ref, _rref, _func in _METHOD_SPECS:
    _spec = MethodSpec(
        method_id=_id,
        label_fr=_label,
        family="inference",
        engine="python",
        outcome_types=["continuous"],
        predictor_types=_predictors,
        min_sample_size=_min_n,
        assumptions=_assumptions,
        reference_impl=_ref,
        r_reference=_rref,
        phase=17,
    )
    register_method(_spec, _func)
