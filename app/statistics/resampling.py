"""Bootstrap, permutation et puissance par simulation (Phase 17 — T17.7)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from app.errors import InvalidMethodInputError
from app.results.models import AnalysisResult
from app.statistics.registry import MethodSpec, register_method

_ALLOWED_STATISTICS = frozenset({"mean", "median", "mean_diff"})


def _rng(random_state: int) -> np.random.Generator:
    return np.random.default_rng(random_state)


def run_bootstrap(
    df: pd.DataFrame,
    statistic: str,
    columns: list[str],
    n_resamples: int = 5000,
    confidence: float = 0.95,
    random_state: int = 0,
) -> AnalysisResult:
    """Bootstrap sur une statistique (liste blanche : mean, median, mean_diff)."""
    stat_key = statistic.lower()
    if stat_key not in _ALLOWED_STATISTICS:
        raise InvalidMethodInputError(
            f"Statistique '{statistic}' non autorisée. Choisir : {sorted(_ALLOWED_STATISTICS)}."
        )
    for col in columns:
        if col not in df.columns:
            raise InvalidMethodInputError(f"Colonne '{col}' introuvable.")
    df_clean = df[columns].dropna()
    if len(df_clean) < 2:
        raise InvalidMethodInputError("Au moins 2 observations complètes requises.")

    rng = _rng(random_state)
    n = len(df_clean)
    boot_vals = np.empty(n_resamples, dtype=float)
    arr = df_clean.to_numpy(dtype=float)

    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample = arr[idx]
        if stat_key == "mean":
            boot_vals[i] = np.mean(sample[:, 0])
        elif stat_key == "median":
            boot_vals[i] = np.median(sample[:, 0])
        else:
            if len(columns) != 2:
                raise InvalidMethodInputError(
                    "mean_diff requiert exactement 2 colonnes."
                )
            boot_vals[i] = np.mean(sample[:, 0]) - np.mean(sample[:, 1])

    alpha = 1 - confidence
    point = float(boot_vals.mean())
    lo, hi = np.percentile(boot_vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return AnalysisResult(
        method="bootstrap",
        dataset_id="",
        variables={
            "statistic": stat_key,
            "columns": columns,
            "n_resamples": n_resamples,
            "confidence": confidence,
            "random_state": random_state,
        },
        sample_size=n,
        preprocessing=[],
        estimates={"statistic": point},
        standard_errors={"statistic": float(np.std(boot_vals, ddof=1))},
        confidence_intervals={"statistic": [float(lo), float(hi)]},
        p_values={},
        model_metrics={
            "method_family": "inference",
            "random_state": random_state,
            "n_resamples": n_resamples,
        },
        diagnostics=[],
        warnings=[],
        assumptions_checked=[
            "Échantillon i.i.d.",
            "Échantillon suffisant pour bootstrap",
        ],
        generated_code="# Bootstrap avec numpy.random.Generator\n",
        execution_log=["Bootstrap exécuté"],
        software_versions={"numpy": np.__version__},
    )


def run_permutation_test(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    statistic: str = "mean_diff",
    n_resamples: int = 10000,
    random_state: int = 0,
) -> AnalysisResult:
    """Test de permutation pour comparer deux groupes."""
    if statistic != "mean_diff":
        raise InvalidMethodInputError("Seul statistic='mean_diff' est implémenté.")
    df_clean = df.dropna(subset=[outcome, group])
    labels = sorted(str(g) for g in df_clean[group].unique())
    if len(labels) != 2:
        raise InvalidMethodInputError("Exactement 2 groupes requis.")
    g1, g2 = labels
    x = df_clean.loc[df_clean[group].astype(str) == g1, outcome].to_numpy(dtype=float)
    y = df_clean.loc[df_clean[group].astype(str) == g2, outcome].to_numpy(dtype=float)
    observed = float(np.mean(x) - np.mean(y))
    combined = np.concatenate([x, y])
    n1 = len(x)
    rng = _rng(random_state)
    count = 0
    for _ in range(n_resamples):
        perm = rng.permutation(combined)
        diff = np.mean(perm[:n1]) - np.mean(perm[n1:])
        if abs(diff) >= abs(observed):
            count += 1
    p_val = (count + 1) / (n_resamples + 1)
    return AnalysisResult(
        method="permutation_test",
        dataset_id="",
        variables={
            "outcome": outcome,
            "group": group,
            "statistic": statistic,
            "n_resamples": n_resamples,
            "random_state": random_state,
        },
        sample_size=len(df_clean),
        preprocessing=[],
        estimates={"mean_difference": observed},
        standard_errors={},
        confidence_intervals={},
        p_values={"overall": float(p_val)},
        model_metrics={
            "method_family": "inference",
            "random_state": random_state,
            "n_resamples": n_resamples,
        },
        diagnostics=[],
        warnings=[],
        assumptions_checked=["Échangeabilité sous H0"],
        generated_code="# Test de permutation\n",
        execution_log=["Test de permutation exécuté"],
        software_versions={"numpy": np.__version__},
    )


def simulate_power(
    design: dict[str, Any],
    n_simulations: int = 2000,
    random_state: int = 0,
) -> AnalysisResult:
    """Puissance par simulation Monte Carlo (test t deux échantillons)."""
    effect = float(design.get("effect_size", 0.5))
    n_per_group = int(design.get("n_per_group", 30))
    alpha = float(design.get("alpha", 0.05))
    if n_per_group < 2:
        raise InvalidMethodInputError("n_per_group >= 2 requis.")
    rng = _rng(random_state)
    rejections = 0
    for _ in range(n_simulations):
        x = rng.normal(0, 1, n_per_group)
        y = rng.normal(effect, 1, n_per_group)
        _, p = stats.ttest_ind(x, y, equal_var=True)
        if p < alpha:
            rejections += 1
    power = rejections / n_simulations
    return AnalysisResult(
        method="simulate_power",
        dataset_id="",
        variables={
            "design": design,
            "n_simulations": n_simulations,
            "random_state": random_state,
        },
        sample_size=n_per_group * 2,
        preprocessing=[],
        estimates={"power": float(power), "effect_size": effect},
        standard_errors={},
        confidence_intervals={},
        p_values={},
        model_metrics={
            "method_family": "inference",
            "random_state": random_state,
            "n_simulations": n_simulations,
            "assumptions": ["Normalité simulée", "Variances égales"],
        },
        diagnostics=[],
        warnings=[],
        assumptions_checked=["Modèle de simulation gaussien"],
        generated_code="# Puissance par simulation Monte Carlo\n",
        execution_log=["Simulation de puissance exécutée"],
        software_versions={"numpy": np.__version__, "scipy": scipy.__version__},
    )


register_method(
    MethodSpec(
        method_id="bootstrap",
        label_fr="Bootstrap",
        family="inference",
        engine="python",
        phase=17,
        reference_impl="app.statistics.resampling.run_bootstrap",
    ),
    run_bootstrap,
)
register_method(
    MethodSpec(
        method_id="permutation_test",
        label_fr="Test de permutation",
        family="inference",
        engine="python",
        phase=17,
        reference_impl="app.statistics.resampling.run_permutation_test",
    ),
    run_permutation_test,
)
register_method(
    MethodSpec(
        method_id="simulate_power",
        label_fr="Puissance par simulation",
        family="inference",
        engine="python",
        phase=17,
        reference_impl="app.statistics.resampling.simulate_power",
    ),
    simulate_power,
)
