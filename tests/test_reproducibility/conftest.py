"""Fixtures partagées pour les tests de reproductibilité."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_dataset():
    """Génère un dataset de test avec plusieurs types de variables.

    Variables :
    - age : numérique (20-65)
    - income : numérique (20000-100000)
    - gender : catégorielle binaire (M/F)
    - education : catégorielle (High School, Bachelor, Master, PhD)
    - satisfaction : catégorielle (Low, Medium, High)
    - score : numérique (0-100)
    """
    np.random.seed(42)
    n = 100

    data = {
        "age": np.random.randint(20, 65, n),
        "income": np.random.randint(20000, 100000, n),
        "gender": np.random.choice(["M", "F"], n),
        "education": np.random.choice(["High School", "Bachelor", "Master", "PhD"], n),
        "satisfaction": np.random.choice(["Low", "Medium", "High"], n),
        "score": np.random.uniform(0, 100, n).round(2),
    }

    return pd.DataFrame(data)


@pytest.fixture
def temp_project_dir(tmp_path):
    """Répertoire temporaire pour les projets de test."""
    return str(tmp_path / "test_projects")
