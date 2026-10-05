"""Implémentation Python du Protocol StatisticalEngineProvider.

Cette classe délègue aux fonctions de app/statistics/ déjà testées.
C'est le seul endroit où le Protocol est concrètement implémenté.
"""

from __future__ import annotations

import pandas as pd

from app.results.models import AnalysisResult
from app.statistics.provider import StatisticalEngineProvider
from app.statistics.descriptive import create_contingency_table
from app.statistics.tests_classiques import run_chi_square, run_t_test, run_anova
from app.statistics.regression import run_linear_regression, run_logistic_regression


class PythonStatisticalEngine(StatisticalEngineProvider):
    """Implémentation Python du Protocol StatisticalEngineProvider.

    Délègue toutes les méthodes aux fonctions déjà testées dans
    app/statistics/*.
    """

    def run_linear_regression(
        self, df: pd.DataFrame, outcome: str, predictors: list[str]
    ) -> AnalysisResult:
        """Régression linéaire multiple (OLS)."""
        return run_linear_regression(df=df, outcome=outcome, predictors=predictors)

    def run_logistic_regression(
        self,
        df: pd.DataFrame,
        outcome: str,
        predictors: list[str],
        reference_category: str | None = None,
    ) -> AnalysisResult:
        """Régression logistique binaire."""
        return run_logistic_regression(
            df=df,
            outcome=outcome,
            predictors=predictors,
            reference_category=reference_category,
        )

    def run_chi_square(
        self, df: pd.DataFrame, var1: str, var2: str
    ) -> AnalysisResult:
        """Test du chi-carré d'indépendance."""
        return run_chi_square(df=df, var1=var1, var2=var2)

    def run_t_test(
        self, df: pd.DataFrame, outcome: str, group: str
    ) -> AnalysisResult:
        """Test t de Student pour 2 groupes."""
        return run_t_test(df=df, outcome=outcome, group=group)

    def run_anova(
        self, df: pd.DataFrame, outcome: str, group: str
    ) -> AnalysisResult:
        """Analyse de variance (ANOVA) à un facteur."""
        return run_anova(df=df, outcome=outcome, group=group)
