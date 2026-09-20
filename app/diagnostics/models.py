"""Modèles de données pour les diagnostics statistiques."""

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
    code: str
    severity: Severity
    message: str
    details: dict = field(default_factory=dict)
