"""Diagnostics pour les tests d'inférence (Phase 17).

Ce module contient les diagnostics déterministes appelés par les tests
paramétriques et non paramétriques. Chaque diagnostic est calculé par du
code déterministe, jamais par un LLM (V1 §2).
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.diagnostics.models import DiagnosticFinding, Severity


def check_equal_variances(
    groups: list[np.ndarray],
    alpha: float = 0.05,
) -> list[DiagnosticFinding]:
    """Vérifie l'homogénéité des variances entre groupes (test de Levene).

    Args:
        groups: Liste de tableaux numpy, un par groupe.
        alpha: Seuil de signification pour le test de Levene.

    Returns:
        Liste de DiagnosticFinding. Retourne UNEQUAL_VARIANCES si p < alpha.

    Note:
        Seuil : p-value du test de Levene < 0.05.
        Source : Ruxton & Beauchamp (2008), "Time for the change: Moving
        to Welch's t-test".
    """
    if len(groups) < 2:
        return []

    # Test de Levene (plus robuste que Bartlett pour les non-normalités)
    statistic, p_value = stats.levene(*groups, center="median")

    findings: list[DiagnosticFinding] = []
    if p_value < alpha:
        findings.append(
            DiagnosticFinding(
                code="UNEQUAL_VARIANCES",
                severity=Severity.PROBABLE_ISSUE,
                message=(
                    f"Les variances entre les groupes sont significativement "
                    f"différentes (test de Levene, p={p_value:.4f}). "
                    f"Envisagez d'utiliser le test t de Welch ou une ANOVA "
                    f"de Welch au lieu du test t classique ou de l'ANOVA standard."
                ),
                details={
                    "test": "levene",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "alpha": alpha,
                },
            )
        )

    return findings


def check_expected_counts(
    contingency_table: pd.DataFrame | np.ndarray,
    threshold: int = 5,
) -> list[DiagnosticFinding]:
    """Vérifie que les effectifs théoriques sont suffisants pour le chi².

    Args:
        contingency_table: Tableau de contingence (observé).
        threshold: Seuil minimal pour les effectifs théoriques (défaut : 5).

    Returns:
        Liste de DiagnosticFinding. Retourne LOW_EXPECTED_COUNTS si plus de
        20 % des cellules ont des effectifs théoriques < threshold.

    Note:
        Règle : pas plus de 20 % des cellules avec effectifs théoriques < 5.
        Source : Cochran (1954), "Some methods for strengthening the common
        chi-squared tests".
    """
    if isinstance(contingency_table, pd.DataFrame):
        observed = contingency_table.values
    else:
        observed = contingency_table

    if observed.size == 0:
        return []

    # Calcul des effectifs théoriques
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    total = observed.sum()

    if total == 0:
        return []

    expected = (row_totals * col_totals) / total

    # Proportion de cellules avec effectifs théoriques < threshold
    low_count_cells = (expected < threshold).sum()
    total_cells = expected.size
    proportion_low = low_count_cells / total_cells

    findings: list[DiagnosticFinding] = []
    if proportion_low > 0.20:
        findings.append(
            DiagnosticFinding(
                code="LOW_EXPECTED_COUNTS",
                severity=Severity.PROBABLE_ISSUE,
                message=(
                    f"{low_count_cells} cellule(s) sur {total_cells} "
                    f"({proportion_low:.1%}) ont des effectifs théoriques "
                    f"inférieurs à {threshold}. Le test du chi² peut ne pas "
                    f"être fiable. Envisagez le test exact de Fisher."
                ),
                details={
                    "low_count_cells": int(low_count_cells),
                    "total_cells": int(total_cells),
                    "proportion_low": float(proportion_low),
                    "threshold": threshold,
                    "min_expected": float(expected.min()),
                },
            )
        )

    return findings


def check_group_sizes(
    groups: list[np.ndarray],
    min_size: int = 10,
) -> list[DiagnosticFinding]:
    """Vérifie que les groupes ont une taille suffisante.

    Args:
        groups: Liste de tableaux numpy, un par groupe.
        min_size: Taille minimale recommandée (défaut : 10).

    Returns:
        Liste de DiagnosticFinding. Retourne SMALL_GROUP_SIZE si un groupe
        a moins de min_size observations.

    Note:
        Seuil : n < 10 par groupe (recommandation courante pour les tests
        paramétriques). Source : pratique courante en statistiques.
    """
    findings: list[DiagnosticFinding] = []

    for i, group in enumerate(groups):
        n = len(group)
        if n < min_size:
            findings.append(
                DiagnosticFinding(
                    code="SMALL_GROUP_SIZE",
                    severity=Severity.WARNING,
                    message=(
                        f"Le groupe {i + 1} ne contient que {n} observation(s). "
                        f"Les tests paramétriques peuvent manquer de puissance "
                        f"avec des groupes de taille inférieure à {min_size}."
                    ),
                    details={
                        "group_index": i,
                        "group_size": n,
                        "min_recommended": min_size,
                    },
                )
            )

    return findings
