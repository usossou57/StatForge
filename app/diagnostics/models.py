"""Modèles pour le moteur de diagnostics statistiques.

Ce module définit les structures de données pour représenter les diagnostics
détectés lors des analyses statistiques.
"""

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    """Niveau de gravité d'un diagnostic — ne jamais transformer un warning en blocage."""

    CERTAIN_ISSUE = "certain_issue"
    PROBABLE_ISSUE = "probable_issue"
    WARNING = "warning"
    INFORMATION = "information"


@dataclass
class DiagnosticFinding:
    """Un diagnostic détecté par le Diagnostic Engine — toujours produit par du code
    déterministe, jamais par un LLM."""

    code: str  # ex. "SEPARATION_COMPLETE", "HIGH_VIF"
    severity: Severity
    message: str  # explication en langage clair (français)
    details: dict = field(default_factory=dict)
