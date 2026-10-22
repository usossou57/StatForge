"""Corrections pour tests multiples et équivalence (Phase 17 — T17.6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from statsmodels.stats.multitest import multipletests

from app.errors import InvalidMethodInputError
from app.results.models import AnalysisResult
from app.statistics.registry import MethodSpec, register_method

_ADJUST_METHODS = {
    "bonferroni": "bonferroni",
    "holm": "holm",
    "fdr_bh": "fdr_bh",
    "bh": "fdr_bh",
    "by": "fdr_by",
    "benjamini-yekutieli": "fdr_by",
}


def adjust_p_values(
    p_values: dict[str, float],
    method: str = "holm",
) -> AnalysisResult:
    """Ajuste un dictionnaire de p-values (Bonferroni, Holm, BH, BY)."""
    if not p_values:
        raise InvalidMethodInputError("Au moins une p-value est requise.")
    key = method.lower().replace(" ", "-")
    sm_method = _ADJUST_METHODS.get(key)
    if sm_method is None:
        raise InvalidMethodInputError(
            f"Méthode '{method}' inconnue. Choisir : bonferroni, holm, fdr_bh, by."
        )
    names = list(p_values.keys())
    raw = np.array([float(p_values[k]) for k in names], dtype=float)
    _, adjusted, _, _ = multipletests(raw, method=sm_method)
    adjusted_dict = {name: float(p_adj) for name, p_adj in zip(names, adjusted)}
    rows = [
        {"comparison": name, "p_raw": float(p_values[name]), "p_adjusted": float(p_adj)}
        for name, p_adj in adjusted_dict.items()
    ]
    return AnalysisResult(
        method="adjust_p_values",
        dataset_id="",
        variables={"method": sm_method, "n_tests": len(names)},
        sample_size=len(names),
        preprocessing=[],
        estimates={},
        standard_errors={},
        confidence_intervals={},
        p_values=adjusted_dict,
        model_metrics={
            "method_family": "inference",
            "adjusted_p_values": adjusted_dict,
            "tables": {"adjustments": rows},
        },
        diagnostics=[],
        warnings=[],
        assumptions_checked=["Indépendance ou dépendance positive (selon méthode)"],
        generated_code=(
            "from statsmodels.stats.multitest import multipletests\n"
            f"# multipletests(pvals, method='{sm_method}')\n"
        ),
        execution_log=["Correction pour tests multiples appliquée"],
        software_versions={"numpy": np.__version__},
    )


def run_tost_equivalence(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    margin: float,
) -> AnalysisResult:
    """Test d'équivalence TOST (deux unilatéraux) pour deux groupes."""
    if outcome not in df.columns or group not in df.columns:
        raise InvalidMethodInputError("Colonnes outcome et group requises.")
    if margin <= 0:
        raise InvalidMethodInputError("La marge d'équivalence doit être > 0.")
    df_clean = df.dropna(subset=[outcome, group])
    labels = sorted(str(g) for g in df_clean[group].unique())
    if len(labels) != 2:
        raise InvalidMethodInputError("TOST : exactement 2 groupes requis.")
    g1, g2 = labels
    x = df_clean.loc[df_clean[group].astype(str) == g1, outcome].to_numpy(dtype=float)
    y = df_clean.loc[df_clean[group].astype(str) == g2, outcome].to_numpy(dtype=float)
    if len(x) < 2 or len(y) < 2:
        raise InvalidMethodInputError("Au moins 2 observations par groupe.")
    diff = float(np.mean(x) - np.mean(y))
    se = float(np.sqrt(np.var(x, ddof=1) / len(x) + np.var(y, ddof=1) / len(y)))
    if se <= 0:
        raise InvalidMethodInputError("Erreur standard nulle : TOST impossible.")
    df_t = len(x) + len(y) - 2
    t_upper = (diff - margin) / se
    t_lower = (diff + margin) / se
    p_upper = float(stats.t.cdf(t_upper, df_t))
    p_lower = float(1 - stats.t.cdf(t_lower, df_t))
    p_equiv = max(p_upper, p_lower)
    return AnalysisResult(
        method="tost_equivalence",
        dataset_id="",
        variables={"outcome": outcome, "group": group, "margin": margin},
        sample_size=len(df_clean),
        preprocessing=[],
        estimates={"mean_difference": diff, "margin": float(margin)},
        standard_errors={"mean_difference": se},
        confidence_intervals={},
        p_values={"overall": p_equiv, "upper": p_upper, "lower": p_lower},
        model_metrics={
            "method_family": "inference",
            "equivalence_decision": p_equiv < 0.05,
            "tables": {
                "tost": [
                    {"test": "upper", "t": t_upper, "p": p_upper},
                    {"test": "lower", "t": t_lower, "p": p_lower},
                ]
            },
        },
        diagnostics=[],
        warnings=[],
        assumptions_checked=[
            "Normalité approximative",
            "Variances homogènes approximatives",
        ],
        generated_code="# TOST : deux tests t unilatéraux\n",
        execution_log=["TOST d'équivalence exécuté"],
        software_versions={"scipy": scipy.__version__},
    )


def run_non_inferiority(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    margin: float,
    alternative: str = "less",
) -> AnalysisResult:
    """Test de non-infériorité (unilatéral) entre deux groupes."""
    if alternative not in {"less", "greater"}:
        raise InvalidMethodInputError("alternative doit être 'less' ou 'greater'.")
    base = run_tost_equivalence(df, outcome, group, margin)
    diff = base.estimates["mean_difference"]
    se = base.standard_errors["mean_difference"]
    df_t = base.sample_size - 2
    if alternative == "less":
        t_stat = (diff + margin) / se
        p_val = float(1 - stats.t.cdf(t_stat, df_t))
    else:
        t_stat = (diff - margin) / se
        p_val = float(stats.t.cdf(t_stat, df_t))
    return AnalysisResult(
        method="non_inferiority",
        dataset_id="",
        variables={
            "outcome": outcome,
            "group": group,
            "margin": margin,
            "alternative": alternative,
        },
        sample_size=base.sample_size,
        preprocessing=[],
        estimates={
            "mean_difference": diff,
            "margin": float(margin),
            "t_statistic": t_stat,
        },
        standard_errors={"mean_difference": se},
        confidence_intervals={},
        p_values={"overall": p_val},
        model_metrics={
            "method_family": "inference",
            "non_inferior": p_val < 0.05,
        },
        diagnostics=[],
        warnings=[],
        assumptions_checked=base.assumptions_checked,
        generated_code="# Test de non-infériorité\n",
        execution_log=["Test de non-infériorité exécuté"],
        software_versions={"scipy": scipy.__version__},
    )


register_method(
    MethodSpec(
        method_id="adjust_p_values",
        label_fr="Correction de p-values multiples",
        family="inference",
        engine="python",
        phase=17,
        reference_impl="statsmodels.stats.multitest.multipletests",
        r_reference="p.adjust",
    ),
    adjust_p_values,
)
register_method(
    MethodSpec(
        method_id="tost_equivalence",
        label_fr="Test d'équivalence (TOST)",
        family="inference",
        engine="python",
        phase=17,
        reference_impl="app.statistics.multiple_testing.run_tost_equivalence",
        r_reference="TOST",
    ),
    run_tost_equivalence,
)
register_method(
    MethodSpec(
        method_id="non_inferiority",
        label_fr="Test de non-infériorité",
        family="inference",
        engine="python",
        phase=17,
        reference_impl="app.statistics.multiple_testing.run_non_inferiority",
    ),
    run_non_inferiority,
)
