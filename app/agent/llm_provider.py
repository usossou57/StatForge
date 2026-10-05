"""Fournisseurs LLM abstraits et implémentations concrètes."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


def _extract_json(text: str) -> str:
    """Extrait le JSON d'une chaîne qui peut contenir des blocs de code markdown."""
    if not text:
        return ""
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


@dataclass
class LLMResponse:
    """Résultat d'un appel LLM."""
    text: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    structured_output: Any | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    """Abstraction agnostique du fournisseur LLM."""
    def complete(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        response_schema: type | None = None,
    ) -> LLMResponse:
        ...


class GroqProvider:
    """Implémentation de LLMProvider utilisant l'API Groq (cloud)."""

    def __init__(
        self,
        model: str = "qwen/qwen3.8-27b",  # Modèle Qwen stable sur Groq
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 500,  # Réduit à 500 pour respecter la limite de 1000 OTPM
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Clé API Groq manquante. Définissez la variable d'environnement GROQ_API_KEY."
            )

        self._client = httpx.Client(
            base_url="https://api.groq.com/openai/v1",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        response_schema: type | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if response_schema:
            payload["response_format"] = {"type": "json_object"}

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self._client.post("/chat/completions", json=payload)
                response.raise_for_status()
                break  # Succès, on sort de la boucle
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    try:
                        error_data = e.response.json()
                        message = error_data.get("error", {}).get("message", "")
                        # Extrait le temps d'attente : "Please try again in 12.405s."
                        match = re.search(r"try again in ([\d.]+)s", message)
                        if match:
                            wait_time = float(match.group(1)) + 1.0
                            print(f"\n[DEBUG] Rate limit (429). Attente de {wait_time:.1f}s (tentative {attempt + 1}/{max_retries})...")
                            time.sleep(wait_time)
                            continue
                    except Exception:
                        pass
                # Si ce n'est pas un 429 ou si on a épuisé les réessais, on lève l'erreur
                print(f"\n[DEBUG GROQ API ERROR] Status: {e.response.status_code}")
                print(f"[DEBUG GROQ API ERROR] Body: {e.response.text}\n")
                raise

        data = response.json()
        choice = data["choices"][0]["message"]
        text = choice.get("content")

        tool_calls = None
        if "tool_calls" in choice and choice["tool_calls"]:
            tool_calls = []
            for tc in choice["tool_calls"]:
                tool_calls.append({
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"]),
                })

        structured_output = None
        if response_schema and text:
            try:
                clean_text = _extract_json(text)
                structured_output = response_schema.model_validate_json(clean_text)
            except Exception:
                pass

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            structured_output=structured_output,
            raw_response=data,
        )


class OllamaProvider:
    """Implémentation de LLMProvider utilisant Ollama (local)."""
    # ... (garder le code OllamaProvider tel qu'il était) ...
    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = httpx.Client(base_url=base_url, timeout=120.0)

    def complete(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        response_schema: type | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if response_schema:
            payload["format"] = "json"

        response = self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        text = data.get("message", {}).get("content")

        structured_output = None
        if response_schema and text:
            try:
                clean_text = _extract_json(text)
                structured_output = response_schema.model_validate_json(clean_text)
            except Exception:
                pass

        return LLMResponse(text=text, tool_calls=None, structured_output=structured_output, raw_response=data)
