"""Protocol StatisticalEngineProvider — abstraction du moteur statistique.

Permet, à terme, d'interchanger un moteur Python et un moteur R (contrainte T6).
Ne jamais appeler statsmodels/R directement en dehors d'une implémentation
de ce Protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from app.results.models import AnalysisResult


@runtime_checkable
class StatisticalEngineProvider(Protocol):
    """Abstraction permettant, à terme, un moteur Python et un moteur R
    interchangeables (C7/T6). Ne jamais appeler statsmodels/R directement
    en dehors d'une implémentation de ce Protocol."""

    def run_linear_regression(
        self, df: pd.DataFrame, outcome: str, predictors: list[str]
    ) -> AnalysisResult:
        """Régression linéaire multiple (OLS)."""
        ...

    def run_logistic_regression(
        self,
        df: pd.DataFrame,
        outcome: str,
        predictors: list[str],
        reference_category: str | None = None,
    ) -> AnalysisResult:
        """Régression logistique binaire."""
        ...

    def run_chi_square(self, df: pd.DataFrame, var1: str, var2: str) -> AnalysisResult:
        """Test du chi-carré d'indépendance."""
        ...

    def run_t_test(self, df: pd.DataFrame, outcome: str, group: str) -> AnalysisResult:
        """Test t de Student pour 2 groupes."""
        ...

    def run_anova(self, df: pd.DataFrame, outcome: str, group: str) -> AnalysisResult:
        """Analyse de variance (ANOVA) à un facteur."""
        ...
