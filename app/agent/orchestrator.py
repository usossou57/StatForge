"""Orchestrateur de conversation LLM <-> Outils.

Ce module gère le flux de conversation entre l'utilisateur, le LLM (via LLMProvider)
et les outils statistiques (via execute_tool). Le LLM propose des actions,
le moteur Python les exécute de manière déterministe (Section 2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.agent.llm_provider import LLMProvider
from app.agent.tools import TOOL_SCHEMAS, execute_tool
from app.results.models import AnalysisResult


@dataclass
class ConversationMessage:
    """Message dans l'historique de conversation."""

    role: str  # "user", "assistant", "system", "tool"
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class OrchestratorResult:
    """Résultat d'un tour de conversation."""

    final_response: str
    analysis_results: list[AnalysisResult] = field(default_factory=list)
    conversation_history: list[ConversationMessage] = field(default_factory=list)
    tool_calls_made: int = 0


class Orchestrator:
    """Orchestrateur de conversation LLM <-> Outils.

    Gère le flux multi-tours : l'utilisateur pose une question, le LLM propose
    des outils à appeler, le moteur Python les exécute, et le LLM interprète
    les résultats pour répondre à l'utilisateur.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        system_prompt: str | None = None,
        max_tool_calls_per_turn: int = 10,
    ) -> None:
        """Initialise l'orchestrateur.

        Args:
            llm_provider: Fournisseur LLM à utiliser (Groq, Ollama, etc.).
            system_prompt: Prompt système optionnel pour guider le LLM.
            max_tool_calls_per_turn: Nombre maximum d'appels d'outils par tour
                (évite les boucles infinies).
        """
        self.llm_provider = llm_provider
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.conversation_history: list[ConversationMessage] = []

    def _default_system_prompt(self) -> str:
        """Retourne le prompt système par défaut."""
        return """Tu es un assistant statistique expert. Tu aides les utilisateurs à analyser des données.

RÈGLES STRICTES :
1. Tu ne calcules JAMAIS directement une statistique (moyenne, coefficient, p-value).
2. Tu utilises TOUJOURS les outils disponibles pour exécuter les analyses.
3. Tu expliques les résultats de manière claire et pédagogique.
4. Tu signales les diagnostics (multicolinéarité, séparation, etc.) détectés par le moteur.
5. Tu génères des rapports et graphiques quand c'est pertinent.

Tu dois répondre en français, de manière concise et précise."""

    def process_user_message(
        self,
        user_message: str,
        df: pd.DataFrame | None = None,
        current_result: AnalysisResult | None = None,
    ) -> OrchestratorResult:
        """Traite un message utilisateur et retourne la réponse finale.

        Gère le flux complet : LLM -> outils -> LLM -> réponse utilisateur.

        Args:
            user_message: Message de l'utilisateur.
            df: DataFrame contenant les données (optionnel).
            current_result: AnalysisResult courant (optionnel, pour générer des rapports).

        Returns:
            OrchestratorResult contenant la réponse finale et l'historique.
        """
        # Ajoute le message utilisateur à l'historique
        self.conversation_history.append(
            ConversationMessage(role="user", content=user_message)
        )

        # Construction des messages pour le LLM
        messages = self._build_messages()

        # Boucle d'appels d'outils
        tool_calls_made = 0
        analysis_results: list[AnalysisResult] = []

        while tool_calls_made < self.max_tool_calls_per_turn:
            # Appel au LLM
            response = self.llm_provider.complete(
                messages=messages,
                tools=TOOL_SCHEMAS,
            )

            # Si pas de tool calls, on a la réponse finale
            if not response.tool_calls or len(response.tool_calls) == 0:
                final_text = response.text or ""
                self.conversation_history.append(
                    ConversationMessage(role="assistant", content=final_text)
                )
                return OrchestratorResult(
                    final_response=final_text,
                    analysis_results=analysis_results,
                    conversation_history=self.conversation_history.copy(),
                    tool_calls_made=tool_calls_made,
                )

            # Exécution des outils
            for tool_call in response.tool_calls:
                tool_calls_made += 1

                # Ajoute le message assistant avec tool calls
                self.conversation_history.append(
                    ConversationMessage(
                        role="assistant",
                        content=response.text,
                        tool_calls=response.tool_calls,
                    )
                )

                # Exécute l'outil
                try:
                    result = execute_tool(
                        name=tool_call["name"],
                        arguments=tool_call["arguments"],
                        df=df,
                        result=current_result,
                    )

                    # Si c'est un AnalysisResult, on le stocke
                    if isinstance(result, AnalysisResult):
                        analysis_results.append(result)
                        current_result = result

                    # Sérialise le résultat pour le LLM
                    result_str = self._serialize_result(result)

                    # Ajoute le message tool
                    self.conversation_history.append(
                        ConversationMessage(
                            role="tool",
                            content=result_str,
                            tool_call_id=tool_call["id"],
                            name=tool_call["name"],
                        )
                    )

                except Exception as e:
                    # En cas d'erreur, on informe le LLM
                    self.conversation_history.append(
                        ConversationMessage(
                            role="tool",
                            content=f"Erreur lors de l'exécution de {tool_call['name']}: {e!s}",
                            tool_call_id=tool_call["id"],
                            name=tool_call["name"],
                        )
                    )

            # Reconstruction des messages pour le prochain tour
            messages = self._build_messages()

        # Si on atteint la limite, on demande une réponse finale sans outils
        final_response = self.llm_provider.complete(messages=messages)
        final_text = final_response.text or "Limite d'appels d'outils atteinte."

        self.conversation_history.append(
            ConversationMessage(role="assistant", content=final_text)
        )

        return OrchestratorResult(
            final_response=final_text,
            analysis_results=analysis_results,
            conversation_history=self.conversation_history.copy(),
            tool_calls_made=tool_calls_made,
        )

    def _build_messages(self) -> list[dict[str, str]]:
        """Construit la liste des messages pour le LLM à partir de l'historique."""
        messages = [{"role": "system", "content": self.system_prompt}]

        for msg in self.conversation_history:
            if msg.role == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "content": msg.content or "",
                        "tool_call_id": msg.tool_call_id or "",
                        "name": msg.name or "",
                    }
                )
            elif msg.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc["arguments"]),
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
            else:
                messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content or "",
                    }
                )

        return messages

    def _serialize_result(self, result: Any) -> str:
        """Sérialise un résultat d'outil en chaîne pour le LLM."""
        if isinstance(result, AnalysisResult):
            # Retourne un résumé structuré
            return json.dumps(
                {
                    "method": result.method,
                    "sample_size": result.sample_size,
                    "estimates": result.estimates,
                    "p_values": result.p_values,
                    "model_metrics": result.model_metrics,
                    "n_diagnostics": len(result.diagnostics),
                    "warnings": result.warnings,
                },
                indent=2,
                ensure_ascii=False,
            )

        if isinstance(result, list):
            # Liste de diagnostics
            return json.dumps(
                [
                    {
                        "code": d.code,
                        "severity": d.severity.value,
                        "message": d.message,
                    }
                    for d in result
                ],
                indent=2,
                ensure_ascii=False,
            )

        if isinstance(result, str):
            return result

        # Fallback
        return str(result)

    def reset_conversation(self) -> None:
        """Réinitialise l'historique de conversation."""
        self.conversation_history.clear()


# ──────────────────────────────────────────────────────────────────────
# Fonctions de correction automatique (Phase 9 — Statistical Agent)
# ──────────────────────────────────────────────────────────────────────

from app.diagnostics.models import DiagnosticFinding, Severity
from app.planner.models import AnalysisPlan
from app.statistics.regression import (
    run_firth_logistic,
)

# Registre des corrections disponibles : code diagnostic -> action de correction
_CORRECTION_RULES: dict[str, dict] = {
    "SEPARATION_COMPLETE": {
        "action": "run_firth_logistic",
        "rationale": (
            "Une séparation complète a été détectée : la régression logistique "
            "standard ne converge pas. La régression de Firth applique une "
            "correction de biais qui garantit la convergence dans ce cas."
        ),
    },
    "QUASI_SEPARATION": {
        "action": "run_firth_logistic",
        "rationale": (
            "Une quasi-séparation a été détectée (erreurs standard anormalement "
            "grandes). La régression de Firth est recommandée pour obtenir des "
            "estimations fiables."
        ),
    },
    "HIGH_MULTICOLLINEARITY": {
        "action": "remove_collinear_predictors",
        "rationale": (
            "Une multicolinéarité sévère (VIF > 10) a été détectée entre les "
            "prédicteurs. Il est recommandé de retirer ou combiner les variables "
            "fortement corrélées avant de relancer l'analyse."
        ),
    },
    "INSUFFICIENT_SAMPLE_SIZE": {
        "action": "request_more_data",
        "rationale": (
            "La taille d'échantillon est insuffisante pour la méthode choisie "
            "(règle empirique : au moins 10 événements par variable prédicteur "
            "pour la régression logistique). Aucun recalcul automatique n'est "
            "possible ; des données supplémentaires sont nécessaires."
        ),
    },
}


def propose_correction(
    diagnostics: list[DiagnosticFinding],
    plan: AnalysisPlan,
) -> dict | None:
    """Propose une correction concrète si un diagnostic critique est présent.

    Parcourt la liste des diagnostics et, si un diagnostic de sévérité
    CERTAIN_ISSUE ou PROBABLE_ISSUE correspond à une règle de correction
    connue, retourne une proposition d'action structurée.

    La proposition contient TOUJOURS requires_confirmation=True (contrainte T14 :
    confirmation humaine obligatoire avant toute exécution).

    Args:
        diagnostics: Liste des diagnostics détectés par le Diagnostic Engine.
        plan: Plan d'analyse courant (pour contextualiser la correction).

    Returns:
        Un dictionnaire structuré contenant :
            - action : nom de l'action de correction (ex: 'run_firth_logistic')
            - rationale : explication en langage clair
            - requires_confirmation : toujours True (T14)
            - original_method : méthode initialement prévue par le plan
        Ou None si aucun diagnostic ne déclenche de correction.
    """
    # On traite les diagnostics par ordre de gravité décroissant
    severity_order = {
        Severity.CERTAIN_ISSUE: 0,
        Severity.PROBABLE_ISSUE: 1,
        Severity.WARNING: 2,
        Severity.INFORMATION: 3,
    }
    sorted_diagnostics = sorted(
        diagnostics,
        key=lambda d: severity_order.get(d.severity, 99),
    )

    for diag in sorted_diagnostics:
        # On ne propose une correction que pour les problèmes certains ou probables
        if diag.severity not in (Severity.CERTAIN_ISSUE, Severity.PROBABLE_ISSUE):
            continue

        if diag.code in _CORRECTION_RULES:
            rule = _CORRECTION_RULES[diag.code]
            return {
                "action": rule["action"],
                "rationale": rule["rationale"],
                "requires_confirmation": True,  # Contrainte T14 : OBLIGATOIRE
                "triggered_by": diag.code,
                "triggered_severity": diag.severity.value,
                "original_method": plan.proposed_method,
                "details": diag.details,
            }

    return None


def execute_confirmed_action(
    action: dict,
    df: pd.DataFrame,
    data_dictionary: DataDictionary,  # noqa: F821 — forward reference
    confirmed: bool = False,
) -> AnalysisResult:
    """Exécute une action de correction APRÈS confirmation explicite de l'utilisateur.

    Cette fonction refuse catégoriquement de s'exécuter si `confirmed` n'est
    pas True (contrainte T14). Toute modification d'état doit être confirmée
    explicitement par l'appelant.

    Args:
        action: Dictionnaire d'action tel que retourné par propose_correction().
            Doit contenir au minimum 'action' et 'requires_confirmation'.
        df: DataFrame contenant les données.
        data_dictionary: Dictionnaire de données du dataset.
        confirmed: Doit être True pour que l'action soit exécutée.

    Returns:
        AnalysisResult contenant le résultat de l'action exécutée.

    Raises:
        ValueError: Si confirmed n'est pas True.
        ValueError: Si l'action n'exige pas de confirmation (sécurité).
        ValueError: Si l'action n'est pas reconnue.
        NotImplementedError: Pour les actions non encore implémentées.
    """
    # Garde-fou T14 : refus catégorique sans confirmation
    if not confirmed:
        raise ValueError(
            "Refus d'exécuter l'action sans confirmation explicite. "
            "L'appelant doit passer confirmed=True après validation humaine."
        )

    # Garde-fou : on vérifie que l'action exigeait bien une confirmation
    if not action.get("requires_confirmation", False):
        raise ValueError(
            "Sécurité : seules les actions marquées requires_confirmation=True "
            "peuvent être exécutées via cette fonction."
        )

    action_name = action.get("action")
    if not action_name:
        raise ValueError("L'action ne contient pas de clé 'action'.")

    # Extraction des paramètres du plan original si disponibles
    original_method = action.get("original_method")

    # Exécution selon le type d'action
    if action_name == "run_firth_logistic":
        # On réutilise les mêmes variables que le plan original
        # Le plan original était une régression logistique, on bascule vers Firth
        # Les variables outcome/predictors doivent être fournies via action['details']
        # ou reconstruites depuis le contexte
        details = action.get("details", {})
        outcome = details.get("outcome")
        predictors = details.get("predictors")

        if not outcome or not predictors:
            raise ValueError(
                "L'action 'run_firth_logistic' nécessite les clés 'outcome' et "
                "'predictors' dans action['details']."
            )

        result = run_firth_logistic(df=df, outcome=outcome, predictors=predictors)

        # Journalisation de l'action dans execution_log (T14 traçabilité)
        result.execution_log.append(
            f"Action de correction exécutée : {action_name} "
            f"(déclenchée par {action.get('triggered_by', 'inconnu')})"
        )
        result.warnings.append(
            f"Correction automatique appliquée : {action.get('rationale', '')}"
        )
        return result

    if action_name == "remove_collinear_predictors":
        raise NotImplementedError(
            "L'action 'remove_collinear_predictors' nécessite une interaction "
            "utilisateur pour choisir quelles variables retirer. "
            "À implémenter via l'interface (Phase 10)."
        )

    if action_name == "request_more_data":
        raise NotImplementedError(
            "L'action 'request_more_data' ne peut pas être exécutée "
            "automatiquement : des données supplémentaires sont nécessaires."
        )

    raise ValueError(
        f"Action '{action_name}' non reconnue. "
        f"Actions disponibles : {', '.join(sorted({r['action'] for r in _CORRECTION_RULES.values()}))}"
    )
