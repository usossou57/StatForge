"""Tailles d'effet et intervalles de confiance (Phase 17 — T17.4)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy
from scipy import stats

from app.results.models import AnalysisResult
from app.statistics.registry import MethodSpec, register_method


def _ci_from_se(estimate: float, se: float, alpha: float = 0.05) -> tuple[float, float]:
    z = stats.norm.ppf(1 - alpha / 2)
    return estimate - z * se, estimate + z * se


def cohens_d(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    """d de Cohen (échantillons indépendants) avec IC 95% approximatif."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return {"value": None, "ci_low": None, "ci_high": None}
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    if pooled <= 0:
        return {"value": 0.0, "ci_low": None, "ci_high": None}
    d = (np.mean(x) - np.mean(y)) / math.sqrt(pooled)
    # Variance approximative (Hedges & Olkin)
    se = math.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    lo, hi = _ci_from_se(d, se)
    return {"value": float(d), "ci_low": float(lo), "ci_high": float(hi)}


def hedges_g(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    """g de Hedges (correction de biais pour petits échantillons)."""
    base = cohens_d(x, y)
    if base["value"] is None:
        return base
    n1, n2 = len(x), len(y)
    df = n1 + n2 - 2
    if df <= 1:
        return base
    j = 1 - 3 / (4 * df - 1)
    g = j * base["value"]
    if base["ci_low"] is not None and base["ci_high"] is not None:
        return {
            "value": float(g),
            "ci_low": float(j * base["ci_low"]),
            "ci_high": float(j * base["ci_high"]),
        }
    return {"value": float(g), "ci_low": None, "ci_high": None}


def eta_squared(ss_effect: float, ss_total: float) -> dict[str, float | None]:
    if ss_total <= 0:
        return {"value": None, "ci_low": None, "ci_high": None}
    val = ss_effect / ss_total
    return {"value": float(val), "ci_low": None, "ci_high": None}


def partial_eta_squared(
    ss_effect: float, ss_effect_plus_error: float
) -> dict[str, float | None]:
    if ss_effect_plus_error <= 0:
        return {"value": None, "ci_low": None, "ci_high": None}
    val = ss_effect / ss_effect_plus_error
    return {"value": float(val), "ci_low": None, "ci_high": None}


def cramers_v(chi2: float, n: int, min_dim: int) -> dict[str, float | None]:
    if n <= 0 or min_dim < 1:
        return {"value": None, "ci_low": None, "ci_high": None}
    val = math.sqrt(chi2 / (n * (min_dim - 1)))
    return {"value": float(val), "ci_low": None, "ci_high": None}


def odds_ratio(a: float, b: float, c: float, d: float) -> dict[str, float | None]:
    """Table 2x2 : lignes (a,b), (c,d). IC log-normal (Woolf)."""
    if min(a, b, c, d) <= 0:
        return {"value": None, "ci_low": None, "ci_high": None}
    or_val = (a * d) / (b * c)
    se_log = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lo = math.exp(math.log(or_val) - 1.96 * se_log)
    hi = math.exp(math.log(or_val) + 1.96 * se_log)
    return {"value": float(or_val), "ci_low": float(lo), "ci_high": float(hi)}


def risk_ratio(a: float, n1: float, c: float, n2: float) -> dict[str, float | None]:
    if n1 <= 0 or n2 <= 0:
        return {"value": None, "ci_low": None, "ci_high": None}
    p1, p2 = a / n1, c / n2
    if p2 <= 0:
        return {"value": None, "ci_low": None, "ci_high": None}
    rr = p1 / p2
    se_log = math.sqrt((1 - p1) / (n1 * p1) + (1 - p2) / (n2 * p2)) if p1 > 0 else None
    if se_log is None:
        return {"value": float(rr), "ci_low": None, "ci_high": None}
    lo = math.exp(math.log(rr) - 1.96 * se_log)
    hi = math.exp(math.log(rr) + 1.96 * se_log)
    return {"value": float(rr), "ci_low": float(lo), "ci_high": float(hi)}


def rank_biserial(u_stat: float, n1: int, n2: int) -> dict[str, float | None]:
    if n1 <= 0 or n2 <= 0:
        return {"value": None, "ci_low": None, "ci_high": None}
    val = 1 - (2 * u_stat) / (n1 * n2)
    return {"value": float(val), "ci_low": None, "ci_high": None}


def _effect_result(
    method: str,
    label: str,
    effect: dict[str, float | None],
    variables: dict[str, Any],
) -> AnalysisResult:
    return AnalysisResult(
        method=method,
        dataset_id="",
        variables=variables,
        sample_size=int(variables.get("n", 0)),
        preprocessing=[],
        estimates={"effect_size": effect.get("value")},
        standard_errors={},
        confidence_intervals={"effect_size": [effect["ci_low"], effect["ci_high"]]}
        if effect.get("ci_low") is not None and effect.get("ci_high") is not None
        else {},
        p_values={},
        model_metrics={
            "method_family": "inference",
            "effect_size": effect,
        },
        diagnostics=[],
        warnings=[],
        assumptions_checked=[],
        generated_code=f"# Taille d'effet : {label}\n",
        execution_log=[f"{label} calculée"],
        software_versions={"numpy": np.__version__, "scipy": scipy.__version__},
    )


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> AnalysisResult:
    effect = cohens_d(x, y)
    return _effect_result(
        "cohens_d",
        "d de Cohen",
        effect,
        {"n": len(x) + len(y)},
    )


register_method(
    MethodSpec(
        method_id="cohens_d",
        label_fr="d de Cohen (deux échantillons)",
        family="inference",
        engine="python",
        outcome_types=["continuous"],
        predictor_types=["binary"],
        min_sample_size=4,
        assumptions=["Indépendance", "Normalité approximative"],
        reference_impl="app.statistics.effect_sizes.cohens_d",
        r_reference="effsize::cohen.d",
        phase=17,
    ),
    compute_cohens_d,
)
