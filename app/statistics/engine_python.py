"""Implémentation concrète du moteur statistique en Python."""

import pandas as pd

from app.results.models import AnalysisResult
from app.statistics.provider import StatisticalEngineProvider
from app.statistics.regression import run_logistic, run_ols


class PythonStatisticalEngine(StatisticalEngineProvider):
    """Implémentation Python du StatisticalEngineProvider.
    
    Cette classe délègue les calculs aux fonctions déterministes de
    app.statistics.regression, respectant ainsi la règle architecturale :
    "L'IA PROPOSE → LE MOTEUR CALCULE → LE VALIDATEUR CONTRÔLE → L'UTILISATEUR DÉCIDE".
    """

    def run_linear_regression(
        self, df: pd.DataFrame, outcome: str, predictors: list[str]
    ) -> AnalysisResult:
        """Délègue la régression linéaire à run_ols."""
        return run_ols(df=df, outcome=outcome, predictors=predictors)

    def run_logistic_regression(
        self,
        df: pd.DataFrame,
        outcome: str,
        predictors: list[str],
        reference_category: str | None = None,
    ) -> AnalysisResult:
        """Délègue la régression logistique à run_logistic."""
        return run_logistic(
            df=df,
            outcome=outcome,
            predictors=predictors,
            reference_category=reference_category,
        )
