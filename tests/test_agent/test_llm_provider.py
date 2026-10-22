"""Tests pour les providers LLM."""

import os
from unittest.mock import MagicMock, patch

import pytest

from app.agent.llm_provider import GroqProvider, LLMResponse, OllamaProvider


def test_llm_response_creation() -> None:
    """Teste la création d'un LLMResponse."""
    response = LLMResponse(
        text="Bonjour",
        tool_calls=None,
        structured_output=None,
    )
    assert response.text == "Bonjour"
    assert response.tool_calls is None
    assert response.structured_output is None


def test_groq_provider_missing_api_key() -> None:
    """Teste que GroqProvider lève une erreur si la clé API est manquante."""
    # Supprime temporairement la variable d'environnement
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Clé API Groq manquante"):
            GroqProvider()


def test_groq_provider_with_api_key() -> None:
    """Teste l'initialisation de GroqProvider avec une clé API explicite."""
    provider = GroqProvider(api_key="test_key_123")
    assert provider.api_key == "test_key_123"
    assert provider.model == "qwen/qwen3.8-27b"


def test_groq_provider_complete_mock() -> None:
    """Teste GroqProvider.complete() avec un mock de l'API."""
    provider = GroqProvider(api_key="test_key")

    # Mock de la réponse HTTP
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Bonjour, comment puis-je vous aider ?",
                    "tool_calls": None,
                }
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(provider._client, "post", return_value=mock_response):
        response = provider.complete(messages=[{"role": "user", "content": "Bonjour"}])

    assert response.text == "Bonjour, comment puis-je vous aider ?"
    assert response.tool_calls is None


def test_groq_provider_complete_with_tool_calls_mock() -> None:
    """Teste GroqProvider.complete() avec des tool calls."""
    provider = GroqProvider(api_key="test_key")

    # Mock de la réponse HTTP avec tool calls
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": {
                                "name": "run_linear_regression",
                                "arguments": '{"outcome": "y", "predictors": ["x1"]}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(provider._client, "post", return_value=mock_response):
        response = provider.complete(
            messages=[{"role": "user", "content": "Fais une régression"}]
        )

    assert response.text is None
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "run_linear_regression"
    assert response.tool_calls[0]["arguments"]["outcome"] == "y"


def test_ollama_provider_initialization() -> None:
    """Teste l'initialisation de OllamaProvider."""
    provider = OllamaProvider(model="llama3.1:8b")
    assert provider.model == "llama3.1:8b"
    assert provider.temperature == 0.3


def test_ollama_provider_complete_mock() -> None:
    """Teste OllamaProvider.complete() avec un mock de l'API."""
    provider = OllamaProvider()

    # Mock de la réponse HTTP
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {
            "content": "Bonjour depuis Ollama !",
        }
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(provider._client, "post", return_value=mock_response):
        response = provider.complete(messages=[{"role": "user", "content": "Bonjour"}])

    assert response.text == "Bonjour depuis Ollama !"
    assert (
        response.tool_calls is None
    )  # Ollama ne supporte pas nativement le tool calling
