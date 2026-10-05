"""Tests pour les schémas d'outils et l'exécuteur."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.agent.tools import TOOL_SCHEMAS, execute_tool
from app.diagnostics.models import DiagnosticFinding
from app.results.models import AnalysisResult


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Crée un DataFrame de test."""
    np.random.seed(42)
    return pd.DataFrame({
        "y": np.random.randn(100),
        "x1": np.random.randn(100),
        "x2": np.random.randn(100),
        "outcome": np.random.choice([0, 1], size=100),
        "group": np.random.choice(["A", "B", "C"], size=100),
    })


@pytest.fixture
def sample_result() -> AnalysisResult:
    """Crée un AnalysisResult de test."""
    return AnalysisResult(
        method="linear_regression",
        dataset_id="test_001",
        variables={"y": {"role": "outcome"}, "x1": {"role": "predictor"}},
        sample_size=100,
        preprocessing=[],
        estimates={"x1": 0.5, "const": 1.0},
        standard_errors={"x1": 0.1, "const": 0.2},
        confidence_intervals={"x1": (0.3, 0.7), "const": (0.6, 1.4)},
        p_values={"x1": 0.001, "const": 0.01},
        model_metrics={"r_squared": 0.5},
        diagnostics=[],
        warnings=[],
        assumptions_checked=[],
        generated_code="# test",
        execution_log=[],
        software_versions={"python": "3.11"},
    )


def test_tool_schemas_structure() -> None:
    """Teste que TOOL_SCHEMAS a la structure OpenAI function calling."""
    assert isinstance(TOOL_SCHEMAS, list)
    assert len(TOOL_SCHEMAS) > 0

    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert "function" in schema
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]


def test_tool_schemas_contain_required_tools() -> None:
    """Teste que les outils principaux sont présents dans TOOL_SCHEMAS."""
    tool_names = [s["function"]["name"] for s in TOOL_SCHEMAS]

    assert "run_linear_regression" in tool_names
    assert "run_logistic_regression" in tool_names
    assert "run_multinomial_logistic_regression" in tool_names
    assert "run_firth_logistic" in tool_names
    assert "check_multicollinearity" in tool_names
    assert "check_sample_size" in tool_names
    assert "generate_report" in tool_names


def test_execute_tool_linear_regression(sample_df: pd.DataFrame) -> None:
    """Teste l'exécution de run_linear_regression via execute_tool."""
    result = execute_tool(
        name="run_linear_regression",
        arguments={"outcome": "y", "predictors": ["x1", "x2"]},
        df=sample_df,
    )

    assert isinstance(result, AnalysisResult)
    assert result.method == "linear_regression"
    assert "x1" in result.estimates


def test_execute_tool_logistic_regression(sample_df: pd.DataFrame) -> None:
    """Teste l'exécution de run_logistic_regression via execute_tool."""
    result = execute_tool(
        name="run_logistic_regression",
        arguments={"outcome": "outcome", "predictors": ["x1"]},
        df=sample_df,
    )

    assert isinstance(result, AnalysisResult)
    assert result.method == "logistic_regression"


def test_execute_tool_check_multicollinearity(sample_df: pd.DataFrame) -> None:
    """Teste l'exécution de check_multicollinearity via execute_tool."""
    result = execute_tool(
        name="check_multicollinearity",
        arguments={"predictors": ["x1", "x2"]},
        df=sample_df,
    )

    assert isinstance(result, list)
    # Sur des données aléatoires, pas de multicolinéarité sévère attendue
    # (mais la liste peut être vide)


def test_execute_tool_check_sample_size(sample_df: pd.DataFrame) -> None:
    """Teste l'exécution de check_sample_size via execute_tool."""
    result = execute_tool(
        name="check_sample_size",
        arguments={"method": "linear_regression", "n_predictors": 2},
        df=sample_df,
    )

    assert isinstance(result, list)
    # 100 obs >= 10 × 2 = 20, donc pas de diagnostic attendu
    assert len(result) == 0


def test_execute_tool_generate_report(sample_result: AnalysisResult) -> None:
    """Teste l'exécution de generate_report via execute_tool."""
    result = execute_tool(
        name="generate_report",
        arguments={"output_format": "markdown"},
        result=sample_result,
    )

    assert isinstance(result, str)
    assert "linear_regression" in result


def test_execute_tool_unknown_tool() -> None:
    """Teste qu'un outil inconnu lève une ValueError."""
    with pytest.raises(ValueError, match="non reconnu"):
        execute_tool(name="unknown_tool", arguments={})


def test_execute_tool_missing_df() -> None:
    """Teste qu'une fonction statistique sans DataFrame lève une ValueError."""
    with pytest.raises(ValueError, match="nécessite un DataFrame"):
        execute_tool(
            name="run_linear_regression",
            arguments={"outcome": "y", "predictors": ["x1"]},
            df=None,
        )


def test_execute_tool_missing_result() -> None:
    """Teste que generate_report sans AnalysisResult lève une ValueError."""
    with pytest.raises(ValueError, match="nécessite un AnalysisResult"):
        execute_tool(
            name="generate_report",
            arguments={"output_format": "html"},
            result=None,
        )
