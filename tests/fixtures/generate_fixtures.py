"""Script de génération des fixtures de test pour la Phase 15.

Ce script crée les datasets de référence utilisés par les tests
d'intégration de la Phase 15. Il peut être exécuté manuellement
pour régénérer les fixtures si nécessaire.

Usage :
    uv run python tests/fixtures/generate_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).parent


def generate_missing_dataset() -> pd.DataFrame:
    """Génère un dataset avec ~30% de valeurs manquantes."""
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame(
        {
            "id": range(n),
            "age": np.random.randint(18, 90, size=n).astype(float),
            "revenu": np.random.uniform(1000, 50000, size=n),
            "score": np.random.uniform(0, 100, size=n),
            "groupe": np.random.choice(["A", "B", "C", "D"], size=n),
            "sexe": np.random.choice(["M", "F"], size=n),
        }
    )
    # Introduire ~30% de manquants aléatoirement
    for col in ["age", "revenu", "score", "groupe"]:
        mask = np.random.random(n) < 0.3
        df.loc[mask, col] = np.nan
    return df


def generate_merge_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Génère deux tables à fusionner avec clés dupliquées."""
    df_left = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 3, 4, 5],
            "visite": ["V1", "V2", "V1", "V1", "V1", "V1"],
            "traitement": ["A", "A", "B", "A", "B", "A"],
        }
    )
    df_right = pd.DataFrame(
        {
            "patient_id": [1, 2, 2, 3, 5, 6],
            "score": [85, 90, 92, 78, 95, 88],
            "date": pd.to_datetime(
                [
                    "2024-01-15",
                    "2024-02-20",
                    "2024-03-25",
                    "2024-04-30",
                    "2024-05-05",
                    "2024-06-10",
                ]
            ),
        }
    )
    return df_left, df_right


def generate_large_csv(path: Path, n_rows: int = 500_000) -> None:
    """Génère un CSV volumineux pour les tests de performance."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "id": range(n_rows),
            "age": np.random.randint(18, 90, size=n_rows),
            "revenu": np.random.uniform(1000, 50000, size=n_rows),
            "score": np.random.uniform(0, 100, size=n_rows),
            "groupe": np.random.choice(["A", "B", "C", "D", "E"], size=n_rows),
        }
    )
    df.to_csv(path, index=False)


def main() -> None:
    """Génère toutes les fixtures."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Dataset avec 30% de manquants
    df_missing = generate_missing_dataset()
    df_missing.to_csv(FIXTURES_DIR / "missing_30pct.csv", index=False)
    print(f"✅ missing_30pct.csv ({len(df_missing)} lignes)")

    # Tables à fusionner
    df_left, df_right = generate_merge_tables()
    df_left.to_csv(FIXTURES_DIR / "merge_left.csv", index=False)
    df_right.to_csv(FIXTURES_DIR / "merge_right.csv", index=False)
    print(f"✅ merge_left.csv ({len(df_left)} lignes)")
    print(f"✅ merge_right.csv ({len(df_right)} lignes)")

    # CSV volumineux (500k lignes)
    large_path = FIXTURES_DIR / "large_500k.csv"
    generate_large_csv(large_path, n_rows=500_000)
    print("✅ large_500k.csv (500 000 lignes)")

    print("\n🎉 Toutes les fixtures ont été générées.")


if __name__ == "__main__":
    main()
