"""Puissance statistique et taille d'échantillon (Phase 17 — T17.5)."""

from __future__ import annotations

from typing import Any

import numpy as np
from statsmodels.stats.power import (
    FTestAnovaPower,
    NormalIndPower,
    TTestIndPower,
)

from app.errors import InvalidMethodInputError
from app.results.models import AnalysisResult
from app.statistics.registry import MethodSpec, register_method


def _base_result(
    method: str,
    variables: dict[str, Any],
    estimates: dict[str, float],
    assumptions: list[str],
) -> AnalysisResult:
    return AnalysisResult(
        method=method,
        dataset_id="",
        variables=variables,
        sample_size=int(estimates.get("n", 0)),
        preprocessing=[],
        estimates=estimates,
        standard_errors={},
        confidence_intervals={},
        p_values={},
        model_metrics={"method_family": "inference", "assumptions": assumptions},
        diagnostics=[],
        warnings=[],
        assumptions_checked=assumptions,
        generated_code="# Calcul de puissance (statsmodels.stats.power)\n",
        execution_log=[f"{method} exécuté"],
        software_versions={"numpy": np.__version__},
    )


def compute_power(
    test: str,
    effect_size: float,
    nobs: int,
    alpha: float = 0.05,
    **kwargs: Any,
) -> AnalysisResult:
    """Puissance pour un test donné (t indépendant, ANOVA F, z proportions)."""
    test = test.lower()
    if test in {"t", "ttest", "welch"}:
        power = TTestIndPower().power(
            effect_size=effect_size,
            nobs1=nobs,
            alpha=alpha,
            ratio=kwargs.get("ratio", 1.0),
        )
    elif test in {"anova", "f"}:
        k_groups = int(kwargs.get("k_groups", 3))
        power = FTestAnovaPower().power(
            effect_size=effect_size, ngroups=k_groups, nobs=nobs, alpha=alpha
        )
    elif test in {"proportion", "z"}:
        power = NormalIndPower().power(
            effect_size=effect_size,
            nobs1=nobs,
            alpha=alpha,
            ratio=kwargs.get("ratio", 1.0),
        )
    else:
        raise InvalidMethodInputError(
            f"Test inconnu : '{test}'. Choisir parmi 't', 'anova', 'proportion'."
        )
    return _base_result(
        "compute_power",
        {
            "test": test,
            "effect_size": effect_size,
            "nobs": nobs,
            "alpha": alpha,
            **kwargs,
        },
        {"power": float(power), "n": float(nobs)},
        ["Hypothèses du test choisi", f"alpha = {alpha}"],
    )


def compute_sample_size(
    test: str,
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    **kwargs: Any,
) -> AnalysisResult:
    test = test.lower()
    if test in {"t", "ttest"}:
        n = TTestIndPower().solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            ratio=kwargs.get("ratio", 1.0),
        )
    elif test in {"anova", "f"}:
        k_groups = int(kwargs.get("k_groups", 3))
        n = FTestAnovaPower().solve_power(
            effect_size=effect_size, ngroups=k_groups, power=power, alpha=alpha
        )
    elif test in {"proportion", "z"}:
        n = NormalIndPower().solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            ratio=kwargs.get("ratio", 1.0),
        )
    else:
        raise InvalidMethodInputError(
            f"Test inconnu : '{test}'. Choisir parmi 't', 'anova', 'proportion'."
        )
    return _base_result(
        "compute_sample_size",
        {
            "test": test,
            "effect_size": effect_size,
            "power": power,
            "alpha": alpha,
            **kwargs,
        },
        {"n_required": float(np.ceil(n)), "n": float(np.ceil(n))},
        ["Hypothèses du test choisi", f"puissance cible = {power}"],
    )


def compute_detectable_effect(
    test: str,
    nobs: int,
    power: float = 0.8,
    alpha: float = 0.05,
    **kwargs: Any,
) -> AnalysisResult:
    test = test.lower()
    if test in {"t", "ttest"}:
        effect = TTestIndPower().solve_power(
            nobs1=nobs, power=power, alpha=alpha, ratio=kwargs.get("ratio", 1.0)
        )
    elif test in {"anova", "f"}:
        k_groups = int(kwargs.get("k_groups", 3))
        effect = FTestAnovaPower().solve_power(
            nobs=nobs, ngroups=k_groups, power=power, alpha=alpha
        )
    elif test in {"proportion", "z"}:
        effect = NormalIndPower().solve_power(
            nobs1=nobs, power=power, alpha=alpha, ratio=kwargs.get("ratio", 1.0)
        )
    else:
        raise InvalidMethodInputError(
            f"Test inconnu : '{test}'. Choisir parmi 't', 'anova', 'proportion'."
        )
    return _base_result(
        "compute_detectable_effect",
        {"test": test, "nobs": nobs, "power": power, "alpha": alpha, **kwargs},
        {"effect_size": float(effect), "n": float(nobs)},
        ["Hypothèses du test choisi", f"puissance cible = {power}"],
    )


from collections.abc import Callable

from app.results.models import AnalysisResult

# Contrat V2 §7.1 : toute méthode enregistrée retourne un AnalysisResult.
MethodCallable = Callable[..., AnalysisResult]

MethodSpecTuple = tuple[
    str,              # method_id
    str,              # label_fr
    MethodCallable,   # fonction implémentée
]

_POWER_SPECS: list[MethodSpecTuple] = [
    ("compute_power", "Puissance statistique", compute_power),
    ("compute_sample_size", "Taille d'échantillon requise", compute_sample_size),
    ("compute_detectable_effect", "Effet détectable", compute_detectable_effect),
]

for _mid, _label, _fn in _POWER_SPECS:
    _spec = MethodSpec(
        method_id=_mid,
        label_fr=_label,
        family="inference",
        engine="python",
        outcome_types=[],
        predictor_types=[],
        min_sample_size=2,
        assumptions=["Modèle statsmodels.stats.power"],
        reference_impl=f"statsmodels.stats.power ({_mid})",
        r_reference="power.t.test",
        phase=17,
    )
    register_method(_spec, _fn)
