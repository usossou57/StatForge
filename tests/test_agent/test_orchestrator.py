"""Tests pour l'orchestrateur de conversation."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.agent.llm_provider import LLMResponse
from app.agent.orchestrator import ConversationMessage, Orchestrator, OrchestratorResult


@pytest.fixture
def mock_llm_provider() -> MagicMock:
    """Crée un mock de LLMProvider."""
    return MagicMock()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Crée un DataFrame de test."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "y": np.random.randn(50),
            "x1": np.random.randn(50),
            "x2": np.random.randn(50),
        }
    )


def test_conversation_message_creation() -> None:
    """Teste la création d'un ConversationMessage."""
    msg = ConversationMessage(role="user", content="Bonjour")
    assert msg.role == "user"
    assert msg.content == "Bonjour"
    assert msg.tool_calls is None


def test_orchestrator_result_creation() -> None:
    """Teste la création d'un OrchestratorResult."""
    result = OrchestratorResult(
        final_response="Voici le résultat",
        analysis_results=[],
        conversation_history=[],
        tool_calls_made=0,
    )
    assert result.final_response == "Voici le résultat"
    assert result.tool_calls_made == 0


def test_orchestrator_initialization(mock_llm_provider: MagicMock) -> None:
    """Teste l'initialisation de l'Orchestrator."""
    orchestrator = Orchestrator(llm_provider=mock_llm_provider)
    assert orchestrator.llm_provider == mock_llm_provider
    assert orchestrator.max_tool_calls_per_turn == 10
    assert "assistant statistique" in orchestrator.system_prompt.lower()


def test_orchestrator_simple_response(mock_llm_provider: MagicMock) -> None:
    """Teste une réponse simple sans appel d'outil."""
    orchestrator = Orchestrator(llm_provider=mock_llm_provider)

    # Mock : le LLM répond directement sans tool calls
    mock_llm_provider.complete.return_value = LLMResponse(
        text="Bonjour ! Comment puis-je vous aider ?",
        tool_calls=None,
    )

    result = orchestrator.process_user_message("Bonjour")

    assert result.final_response == "Bonjour ! Comment puis-je vous aider ?"
    assert result.tool_calls_made == 0
    assert len(result.conversation_history) == 2  # user + assistant


def test_orchestrator_with_tool_call(
    mock_llm_provider: MagicMock, sample_df: pd.DataFrame
) -> None:
    """Teste un tour de conversation avec un appel d'outil."""
    orchestrator = Orchestrator(llm_provider=mock_llm_provider)

    # Premier appel : le LLM demande un tool call
    first_response = LLMResponse(
        text=None,
        tool_calls=[
            {
                "id": "call_123",
                "name": "run_linear_regression",
                "arguments": {"outcome": "y", "predictors": ["x1"]},
            }
        ],
    )

    # Deuxième appel : le LLM interprète le résultat
    second_response = LLMResponse(
        text="La régression linéaire montre un coefficient significatif pour x1.",
        tool_calls=None,
    )

    mock_llm_provider.complete.side_effect = [first_response, second_response]

    result = orchestrator.process_user_message(
        "Fais une régression linéaire de y sur x1", df=sample_df
    )

    assert "régression linéaire" in result.final_response.lower()
    assert result.tool_calls_made == 1
    assert len(result.analysis_results) == 1
    assert result.analysis_results[0].method == "linear_regression"



def test_orchestrator_tool_error_handling(
    mock_llm_provider: MagicMock, sample_df: pd.DataFrame
) -> None:
    """Teste la gestion d'erreur lors d'un appel d'outil."""
    orchestrator = Orchestrator(llm_provider=mock_llm_provider)

    # Le LLM demande un outil avec des arguments invalides
    first_response = LLMResponse(
        text=None,
        tool_calls=[
            {
                "id": "call_456",
                "name": "run_linear_regression",
                "arguments": {"outcome": "nonexistent", "predictors": ["x1"]},
            }
        ],
    )

    # Le LLM explique l'erreur
    second_response = LLMResponse(
        text="Désolé, la variable 'nonexistent' n'existe pas dans le dataset.",
        tool_calls=None,
    )

    mock_llm_provider.complete.side_effect = [first_response, second_response]

    result = orchestrator.process_user_message("Fais une régression", df=sample_df)

    assert (
        "nonexistent" in result.final_response
        or "erreur" in result.final_response.lower()
    )
    assert result.tool_calls_made == 1


def test_orchestrator_max_tool_calls_limit(mock_llm_provider: MagicMock) -> None:
    """Teste la limite d'appels d'outils par tour."""
    orchestrator = Orchestrator(
        llm_provider=mock_llm_provider, max_tool_calls_per_turn=2
    )

    # Le LLM demande toujours des tool calls
    tool_response = LLMResponse(
        text=None,
        tool_calls=[
            {
                "id": "call_789",
                "name": "check_sample_size",
                "arguments": {"method": "linear_regression", "n_predictors": 1},
            }
        ],
    )

    mock_llm_provider.complete.return_value = tool_response

    result = orchestrator.process_user_message("Vérifie la taille d'échantillon")

    # Doit s'arrêter après 2 appels
    assert result.tool_calls_made == 2


def test_orchestrator_reset_conversation(mock_llm_provider: MagicMock) -> None:
    """Teste la réinitialisation de la conversation."""
    orchestrator = Orchestrator(llm_provider=mock_llm_provider)

    # Configure le mock pour retourner une réponse simple (sans tool calls)
    mock_llm_provider.complete.return_value = LLMResponse(
        text="Bonjour !",
        tool_calls=None,
    )

    # Ajoute des messages
    orchestrator.process_user_message("Bonjour")
    assert len(orchestrator.conversation_history) > 0

    # Réinitialise
    orchestrator.reset_conversation()
    assert len(orchestrator.conversation_history) == 0


def test_orchestrator_custom_system_prompt(mock_llm_provider: MagicMock) -> None:
    """Teste l'utilisation d'un prompt système personnalisé."""
    custom_prompt = "Tu es un expert en statistiques bayésiennes."
    orchestrator = Orchestrator(
        llm_provider=mock_llm_provider, system_prompt=custom_prompt
    )

    assert orchestrator.system_prompt == custom_prompt


def test_orchestrator_serialize_analysis_result(mock_llm_provider: MagicMock) -> None:
    """Teste la sérialisation d'un AnalysisResult."""
    orchestrator = Orchestrator(llm_provider=mock_llm_provider)

    from app.results.models import AnalysisResult

    result = AnalysisResult(
        method="linear_regression",
        dataset_id="test",
        variables={"y": {"role": "outcome"}},
        sample_size=100,
        preprocessing=[],
        estimates={"x1": 0.5},
        standard_errors={"x1": 0.1},
        confidence_intervals={"x1": (0.3, 0.7)},
        p_values={"x1": 0.001},
        model_metrics={"r_squared": 0.5},
        diagnostics=[],
        warnings=[],
        assumptions_checked=[],
        generated_code="# test",
        execution_log=[],
        software_versions={},
    )

    serialized = orchestrator._serialize_result(result)
    assert "linear_regression" in serialized
    assert "0.5" in serialized  # estimate
    assert "0.001" in serialized  # p-value


# ──────────────────────────────────────────────────────────────────────
# Tests pour propose_correction et execute_confirmed_action (Phase 9)
# ──────────────────────────────────────────────────────────────────────

import pandas as pd

from app.agent.orchestrator import execute_confirmed_action, propose_correction
from app.diagnostics.models import DiagnosticFinding, Severity
from app.planner.models import AnalysisPlan


@pytest.fixture
def sample_plan() -> AnalysisPlan:
    """Crée un AnalysisPlan de test."""
    return AnalysisPlan(
        research_question="Test",
        outcome_variable="outcome",
        predictor_variables=["x1", "x2"],
        study_design="predictive",
        proposed_method="logistic_regression",
        assumptions_to_check=["Indépendance"],
        rationale="Test",
        confidence="high",
    )


@pytest.fixture
def sample_df_for_correction() -> pd.DataFrame:
    """Crée un DataFrame pour tester execute_confirmed_action."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "outcome": np.random.choice([0, 1], size=100),
            "x1": np.random.randn(100),
            "x2": np.random.randn(100),
        }
    )


def test_propose_correction_returns_none_on_clean_diagnostics(
    sample_plan: AnalysisPlan,
) -> None:
    """Teste qu'aucune correction n'est proposée si les diagnostics sont bénins."""
    diagnostics = [
        DiagnosticFinding(
            code="MINOR_ISSUE",
            severity=Severity.INFORMATION,
            message="Information mineure",
        ),
    ]
    result = propose_correction(diagnostics, sample_plan)
    assert result is None


def test_propose_correction_detects_separation(
    sample_plan: AnalysisPlan,
) -> None:
    """Teste qu'une séparation complète déclenche la correction Firth."""
    diagnostics = [
        DiagnosticFinding(
            code="SEPARATION_COMPLETE",
            severity=Severity.CERTAIN_ISSUE,
            message="Séparation complète détectée",
            details={"outcome": "outcome", "predictors": ["x1", "x2"]},
        ),
    ]
    result = propose_correction(diagnostics, sample_plan)

    assert result is not None
    assert result["action"] == "run_firth_logistic"
    assert result["requires_confirmation"] is True  # T14
    assert result["triggered_by"] == "SEPARATION_COMPLETE"
    assert result["original_method"] == "logistic_regression"
    assert "Firth" in result["rationale"]


def test_propose_correction_prioritizes_certainty(
    sample_plan: AnalysisPlan,
) -> None:
    """Teste que CERTAIN_ISSUE est prioritaire sur PROBABLE_ISSUE."""
    diagnostics = [
        DiagnosticFinding(
            code="QUASI_SEPARATION",
            severity=Severity.PROBABLE_ISSUE,
            message="Quasi-séparation",
        ),
        DiagnosticFinding(
            code="SEPARATION_COMPLETE",
            severity=Severity.CERTAIN_ISSUE,
            message="Séparation complète",
        ),
    ]
    result = propose_correction(diagnostics, sample_plan)

    assert result is not None
    assert result["triggered_by"] == "SEPARATION_COMPLETE"
    assert result["triggered_severity"] == "certain_issue"


def test_propose_correction_ignores_warnings(
    sample_plan: AnalysisPlan,
) -> None:
    """Teste que les WARNINGs ne déclenchent pas de correction automatique."""
    diagnostics = [
        DiagnosticFinding(
            code="SEPARATION_COMPLETE",
            severity=Severity.WARNING,  # Sévérité trop faible
            message="Séparation suspectée",
        ),
    ]
    result = propose_correction(diagnostics, sample_plan)
    assert result is None


def test_execute_confirmed_action_requires_confirmation(
    sample_plan: AnalysisPlan,
) -> None:
    """Teste que l'exécution refuse catégoriquement sans confirmation (T14)."""
    action = {
        "action": "run_firth_logistic",
        "requires_confirmation": True,
        "details": {"outcome": "outcome", "predictors": ["x1"]},
    }

    with pytest.raises(ValueError, match="confirmation explicite"):
        execute_confirmed_action(
            action=action,
            df=pd.DataFrame(),
            data_dictionary=None,
            confirmed=False,
        )


def test_execute_confirmed_action_requires_confirmation_flag_in_action() -> None:
    """Teste qu'une action sans requires_confirmation=True est refusée."""
    action = {
        "action": "run_firth_logistic",
        "requires_confirmation": False,  # Drapeau manquant
        "details": {"outcome": "outcome", "predictors": ["x1"]},
    }

    with pytest.raises(ValueError, match="requires_confirmation=True"):
        execute_confirmed_action(
            action=action,
            df=pd.DataFrame(),
            data_dictionary=None,
            confirmed=True,
        )


def test_execute_confirmed_action_runs_firth(
    sample_df_for_correction: pd.DataFrame,
) -> None:
    """Teste l'exécution complète de la correction Firth après confirmation."""
    action = {
        "action": "run_firth_logistic",
        "requires_confirmation": True,
        "triggered_by": "SEPARATION_COMPLETE",
        "rationale": "Test de correction",
        "details": {"outcome": "outcome", "predictors": ["x1"]},
    }

    result = execute_confirmed_action(
        action=action,
        df=sample_df_for_correction,
        data_dictionary=None,
        confirmed=True,
    )

    assert result.method == "firth_logistic_regression"
    # Vérifie que l'action a été journalisée (T14 traçabilité)
    assert any("correction" in log.lower() for log in result.execution_log)
    assert any("Correction automatique" in w for w in result.warnings)


def test_execute_confirmed_action_unknown_action() -> None:
    """Teste qu'une action inconnue lève une ValueError."""
    action = {
        "action": "unknown_action",
        "requires_confirmation": True,
    }

    with pytest.raises(ValueError, match="non reconnue"):
        execute_confirmed_action(
            action=action,
            df=pd.DataFrame(),
            data_dictionary=None,
            confirmed=True,
        )


def test_end_to_end_separation_to_firth(
    sample_df_for_correction: pd.DataFrame,
    sample_plan: AnalysisPlan,
) -> None:
    """Scénario complet : séparation détectée -> proposition Firth -> confirmation -> exécution."""
    # 1. Diagnostic détecté
    diagnostics = [
        DiagnosticFinding(
            code="SEPARATION_COMPLETE",
            severity=Severity.CERTAIN_ISSUE,
            message="Séparation complète",
            details={"outcome": "outcome", "predictors": ["x1"]},
        ),
    ]

    # 2. Proposition de correction
    proposal = propose_correction(diagnostics, sample_plan)
    assert proposal is not None
    assert proposal["requires_confirmation"] is True

    # 3. Exécution après confirmation
    result = execute_confirmed_action(
        action=proposal,
        df=sample_df_for_correction,
        data_dictionary=None,
        confirmed=True,
    )

    assert result.method == "firth_logistic_regression"
    assert result.sample_size == 100
