"""Journal de transformations d'un dataset (V2 §7.2).

Ce module fournit les classes TransformStep et TransformLog qui permettent
de tracer, sérialiser et rejouer toutes les transformations appliquées à
un DataFrame pandas. Chaque transformation est enregistrée avec son code
Python reproductible, garantissant la traçabilité complète du pipeline
de préparation des données.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def _datetime_to_iso(dt: datetime) -> str:
    """Convertit un datetime en chaîne ISO 8601."""
    return dt.isoformat()


def _iso_to_datetime(iso_str: str) -> datetime:
    """Convertit une chaîne ISO 8601 en datetime."""
    return datetime.fromisoformat(iso_str)


@dataclass
class TransformStep:
    """Une opération appliquée à un DataFrame, rejouable.

    Attributs:
        step_id: Identifiant unique de l'étape (ex: 'step_001').
        operation: Type d'opération (ex: 'recode', 'standardize',
            'impute_median').
        params: Paramètres de l'opération (dictionnaire libre).
        columns_in: Colonnes en entrée de l'opération.
        columns_out: Colonnes en sortie de l'opération.
        n_rows_before: Nombre de lignes avant l'opération.
        n_rows_after: Nombre de lignes après l'opération.
        code: Extrait Python reproduisant l'opération sur un DataFrame
            nommé 'df'.
        created_at: Horodatage UTC de la création de l'étape.
    """

    step_id: str
    operation: str
    params: dict[str, Any]
    columns_in: list[str]
    columns_out: list[str]
    n_rows_before: int
    n_rows_after: int
    code: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class TransformLog:
    """Journal chronologique des transformations d'un dataset.

    Attributs:
        steps: Liste ordonnée des étapes de transformation appliquées.
    """

    steps: list[TransformStep] = field(default_factory=list)

    def add(self, step: TransformStep) -> None:
        """Ajoute une étape de transformation au journal.

        Args:
            step: L'étape de transformation à enregistrer.
        """
        self.steps.append(step)

    def to_json(self) -> str:
        """Sérialise le journal en chaîne JSON.

        Les datetimes sont convertis en format ISO 8601 pour être
        JSON-sérialisables.

        Returns:
            Chaîne JSON représentant le journal complet.
        """
        serializable_steps = []
        for step in self.steps:
            step_dict = asdict(step)
            step_dict["created_at"] = _datetime_to_iso(step.created_at)
            serializable_steps.append(step_dict)
        return json.dumps(
            {"steps": serializable_steps},
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str) -> TransformLog:
        """Désérialise un journal depuis une chaîne JSON.

        Args:
            json_str: Chaîne JSON produite par to_json().

        Returns:
            Une instance de TransformLog reconstituée.
        """
        data = json.loads(json_str)
        steps = []
        for step_dict in data["steps"]:
            step_dict["created_at"] = _iso_to_datetime(step_dict["created_at"])
            steps.append(TransformStep(**step_dict))
        return cls(steps=steps)

    def to_python_script(self) -> str:
        """Génère un script Python autonome rejouant toutes les étapes.

        Le script produit charge un DataFrame nommé 'df' (à partir d'un
        fichier CSV dont le chemin est à compléter), puis exécute
        séquentiellement le code de chaque étape enregistrée.

        Returns:
            Un script Python complet et autonome sous forme de chaîne.
        """
        steps_code_parts = []
        for i, step in enumerate(self.steps, start=1):
            steps_code_parts.append(
                f"    # --- Étape {i} : {step.operation} ({step.step_id}) ---\n"
                f"    # Colonnes en entrée : {step.columns_in}\n"
                f"    # Colonnes en sortie : {step.columns_out}\n"
                f"    # Lignes avant : {step.n_rows_before}, après : {step.n_rows_after}\n"
                f"    {step.code}\n"
            )

        steps_block = "\n".join(steps_code_parts)

        # Construction du script sans indentation parasite
        lines = [
            "#!/usr/bin/env python3",
            '"""Script reproductible — Journal de transformations.',
            "Généré automatiquement par StatForge (TransformLog).",
            f"Rejoue {len(self.steps)} étape(s) de transformation.",
            '"""',
            "import pandas as pd",
            "",
            "",
            "def main() -> None:",
            "    # TODO: Remplacer par le chemin réel du fichier de données",
            '    df = pd.read_csv("donnees.csv")',
            '    print(f"Dataset initial : {df.shape[0]} lignes, {df.shape[1]} colonnes")',
            "",
            steps_block,
            '    print(f"\\nDataset final : {df.shape[0]} lignes, {df.shape[1]} colonnes")',
            '    print("Toutes les transformations ont été rejouées.")',
            "",
            "",
            'if __name__ == "__main__":',
            "    main()",
            "",
        ]

        return "\n".join(lines)
