"""Modèles de données pour les résultats d'analyse statistique.

Ce module définit la structure AnalysisResult, objet structuré unique
représentant le résultat complet d'une analyse statistique.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.diagnostics.models import DiagnosticFinding


@dataclass
class AnalysisResult:
    """Objet structuré unique représentant le résultat d'une analyse —
    jamais remplacé par un simple tableau affiché à l'écran.

    Cet objet contient l'intégralité des informations produites par le
    moteur statistique : estimations, erreurs standards, intervalles de
    confiance, p-values, métriques du modèle, diagnostics et avertissements.

    Attributes:
        method: Nom de la méthode statistique utilisée.
        dataset_id: Identifiant du jeu de données analysé.
        variables: Dictionnaire décrivant les variables utilisées.
        sample_size: Taille de l'échantillon après prétraitement.
        preprocessing: Liste des étapes de prétraitement appliquées.
        estimates: Dictionnaire des estimations (coefficients, etc.).
        standard_errors: Dictionnaire des erreurs standards.
        confidence_intervals: Dictionnaire des intervalles de confiance.
        p_values: Dictionnaire des p-values.
        model_metrics: Dictionnaire des métriques du modèle (R², AIC, etc.).
        diagnostics: Liste des diagnostics effectués.
        warnings: Liste des avertissements générés durant l'analyse.
        assumptions_checked: Liste des hypothèses vérifiées.
        generated_code: Code Python généré pour reproduire l'analyse.
        execution_log: Journal d'exécution de l'analyse.
        software_versions: Versions des logiciels/bibliothèques utilisés.
        created_at: Horodatage de création du résultat (UTC).
    """

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
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def to_json(self) -> str:
        """Sérialise le résultat d'analyse en JSON.

        Returns:
            Chaîne JSON représentant l'intégralité du résultat.
        """
        import json
        from dataclasses import asdict

        return json.dumps(asdict(self), default=str, ensure_ascii=False)
