"""Modèles pour les résultats d'analyses statistiques.
Ce module définit la structure AnalysisResult, objet central représentant
le résultat complet d'une analyse statistique.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.diagnostics.models import DiagnosticFinding


@dataclass
class AnalysisResult:
    """Objet structuré unique représentant le résultat d'une analyse — jamais
    remplacé par un simple tableau affiché à l'écran."""

    method: str
    dataset_id: str
    variables: dict
    sample_size: int
    preprocessing: list[str]
    estimates: dict
    standard_errors: dict
    confidence_intervals: dict
    p_values: dict
    model_metrics: dict
    diagnostics: list[DiagnosticFinding]
    warnings: list[str]
    assumptions_checked: list[str]
    generated_code: str
    execution_log: list[str]
    software_versions: dict
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    project_id: str | None = None  # ← AJOUT : lien vers le projet sauvegardé

    def to_json(self) -> str:
        """Convertit l'AnalysisResult en chaîne JSON.
        Returns:
            Représentation JSON de l'objet.
        """
        data = {
            "method": self.method,
            "dataset_id": self.dataset_id,
            "variables": self.variables,
            "sample_size": self.sample_size,
            "preprocessing": self.preprocessing,
            "estimates": self.estimates,
            "standard_errors": self.standard_errors,
            "confidence_intervals": self.confidence_intervals,
            "p_values": self.p_values,
            "model_metrics": self.model_metrics,
            "diagnostics": [
                {
                    "code": d.code,
                    "severity": d.severity.value,
                    "message": d.message,
                    "details": d.details,
                }
                for d in self.diagnostics
            ],
            "warnings": self.warnings,
            "assumptions_checked": self.assumptions_checked,
            "generated_code": self.generated_code,
            "execution_log": self.execution_log,
            "software_versions": self.software_versions,
            "created_at": self.created_at.isoformat(),
        }
        if self.project_id is not None:
            data["project_id"] = self.project_id

        return json.dumps(data, indent=2, ensure_ascii=False)
