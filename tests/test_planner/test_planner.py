"""Tests pour le planner d'analyse."""

from unittest.mock import MagicMock

import pytest

from app.agent.llm_provider import LLMResponse
from app.data.models import DataDictionary, Variable
from app.planner.models import AnalysisPlan
from app.planner.planner import build_analysis_plan, needs_clarification


@pytest.fixture
def sample_data_dictionary() -> DataDictionary:
    """Crée un DataDictionary de test."""
    return DataDictionary(
        dataset_name="test_dataset",
        n_rows=100,
        n_columns=5,
        variables={
            "age": Variable(name="age", dtype="numeric", semantic_type="continuous"),
            "income": Variable(
                name="income", dtype="numeric", semantic_type="continuous"
            ),
            "gender": Variable(
                name="gender",
                dtype="categorical",
                semantic_type="binary",
                levels=["M", "F"],
            ),
            "education": Variable(
                name="education",
                dtype="categorical",
                semantic_type="ordinal",
                levels=["High School", "Bachelor", "Master", "PhD"],
            ),
            "satisfaction": Variable(
                name="satisfaction",
                dtype="categorical",
                semantic_type="ordinal",
                levels=["Low", "Medium", "High"],
            ),
        },
        source_path="/fake/path.csv",
        file_hash="abc123",
    )


def test_analysis_plan_creation() -> None:
    """Teste la création d'un AnalysisPlan."""
    plan = AnalysisPlan(
        research_question="Quel est l'effet de X sur Y ?",
        outcome_variable="Y",
        predictor_variables=["X", "Z"],
        study_design="predictive",
        proposed_method="linear_regression",
        reference_category=None,
        assumptions_to_check=["Linéarité", "Normalité"],
        rationale="Y est numérique continu.",
        confidence="high",
    )

    assert plan.research_question == "Quel est l'effet de X sur Y ?"
    assert plan.outcome_variable == "Y"
    assert plan.proposed_method == "linear_regression"
    assert plan.confidence == "high"


def test_needs_clarification_low_confidence() -> None:
    """Teste que needs_clarification retourne True pour confidence='low'."""
    plan = AnalysisPlan(
        research_question="Question ambiguë",
        outcome_variable="Y",
        predictor_variables=["X"],
        study_design="associative",
        proposed_method="linear_regression",
        assumptions_to_check=[],
        rationale="Pas clair.",
        confidence="low",
    )

    assert needs_clarification(plan) is True


def test_needs_clarification_high_confidence() -> None:
    """Teste que needs_clarification retourne False pour confidence='high'."""
    plan = AnalysisPlan(
        research_question="Question claire",
        outcome_variable="Y",
        predictor_variables=["X"],
        study_design="predictive",
        proposed_method="linear_regression",
        assumptions_to_check=["Linéarité"],
        rationale="Clair.",
        confidence="high",
    )

    assert needs_clarification(plan) is False


def test_build_analysis_plan_with_mock_llm(
    sample_data_dictionary: DataDictionary,
) -> None:
    """Teste build_analysis_plan avec un mock de LLM."""
    mock_llm = MagicMock()

    # Mock : le LLM retourne un plan structuré valide
    mock_plan = AnalysisPlan(
        research_question="Quel est l'effet de l'âge sur le revenu ?",
        outcome_variable="income",
        predictor_variables=["age", "education"],
        study_design="predictive",
        proposed_method="linear_regression",
        reference_category=None,
        assumptions_to_check=["Linéarité", "Normalité des résidus", "Homoscédasticité"],
        rationale="income est numérique continu, une régression linéaire est appropriée.",
        confidence="high",
    )

    mock_llm.complete.return_value = LLMResponse(
        text=None,
        structured_output=mock_plan,
    )

    plan = build_analysis_plan(
        question="Quel est l'effet de l'âge sur le revenu ?",
        data_dictionary=sample_data_dictionary,
        llm=mock_llm,
    )

    assert plan.outcome_variable == "income"
    assert plan.proposed_method == "linear_regression"
    assert plan.confidence == "high"


def test_build_analysis_plan_with_json_text(
    sample_data_dictionary: DataDictionary,
) -> None:
    """Teste build_analysis_plan quand le LLM retourne du JSON texte."""
    mock_llm = MagicMock()

    # Mock : le LLM retourne du JSON texte
    json_text = """{
        "research_question": "Test",
        "outcome_variable": "age",
        "predictor_variables": ["income"],
        "study_design": "associative",
        "proposed_method": "linear_regression",
        "reference_category": null,
        "assumptions_to_check": ["Linéarité"],
        "rationale": "Test rationale",
        "confidence": "medium"
    }"""

    mock_llm.complete.return_value = LLMResponse(
        text=json_text,
        structured_output=None,
    )

    plan = build_analysis_plan(
        question="Test",
        data_dictionary=sample_data_dictionary,
        llm=mock_llm,
    )

    assert plan.outcome_variable == "age"
    assert plan.confidence == "medium"


def test_build_analysis_plan_invalid_json(
    sample_data_dictionary: DataDictionary,
) -> None:
    """Teste que build_analysis_plan lève une erreur sur JSON invalide."""
    mock_llm = MagicMock()

    # Mock : le LLM retourne du texte invalide
    mock_llm.complete.return_value = LLMResponse(
        text="Ce n'est pas du JSON valide",
        structured_output=None,
    )

    with pytest.raises(ValueError, match="plan d'analyse valide"):
        build_analysis_plan(
            question="Test",
            data_dictionary=sample_data_dictionary,
            llm=mock_llm,
        )


def test_build_analysis_plan_no_response(
    sample_data_dictionary: DataDictionary,
) -> None:
    """Teste que build_analysis_plan lève une erreur si le LLM ne retourne rien."""
    mock_llm = MagicMock()

    # Mock : le LLM ne retourne rien
    mock_llm.complete.return_value = LLMResponse(
        text=None,
        structured_output=None,
    )

    with pytest.raises(ValueError, match="ni texte ni output structuré"):
        build_analysis_plan(
            question="Test",
            data_dictionary=sample_data_dictionary,
            llm=mock_llm,
        )
