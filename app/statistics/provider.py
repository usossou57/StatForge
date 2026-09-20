"""Abstraction du moteur statistique."""

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
    ) -> AnalysisResult: ...

    def run_logistic_regression(
        self,
        df: pd.DataFrame,
        outcome: str,
        predictors: list[str],
        reference_category: str | None = None,
    ) -> AnalysisResult: ...
