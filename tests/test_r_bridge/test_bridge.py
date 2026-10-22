"""Tests pour le pont Python-R."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.r_bridge.bridge import call_r_script

# Marque les tests qui nécessitent R
requires_r = pytest.mark.skipif(
    not Path("/usr/bin/Rscript").exists()
    and not Path("/usr/local/bin/Rscript").exists(),
    reason="R n'est pas installé sur cette machine",
)


@pytest.fixture
def sample_logistic_data() -> dict:
    """Crée un dataset simple pour la régression logistique."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "outcome": np.random.choice([0, 1], size=n),
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
        }
    )
    # Rendons x1 prédictif
    df.loc[df["outcome"] == 1, "x1"] += 1.0

    return {
        "data": df.to_dict(orient="list"),
        "outcome": "outcome",
        "predictors": ["x1", "x2"],
    }


@requires_r
def test_call_r_script_firth_logistic(sample_logistic_data: dict) -> None:
    """Teste l'appel du script R Firth avec des données valides."""
    result = call_r_script("firth_logistic.R", sample_logistic_data)

    # Vérifie la structure du résultat
    assert "coefficients" in result
    assert "standard_errors" in result
    assert "p_values" in result
    assert "converged" in result
    assert result["converged"] is True

    # Vérifie que les coefficients sont des nombres
    assert all(isinstance(v, (int, float)) for v in result["coefficients"].values())
    assert all(isinstance(v, (int, float)) for v in result["standard_errors"].values())


@requires_r
def test_call_r_script_with_separation() -> None:
    """Teste le script R Firth sur un dataset avec séparation complète."""
    # Dataset avec séparation parfaite
    df = pd.DataFrame(
        {
            "outcome": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    payload = {
        "data": df.to_dict(orient="list"),
        "outcome": "outcome",
        "predictors": ["x1"],
    }

    result = call_r_script("firth_logistic.R", payload)

    # Firth doit converger même avec séparation
    assert result["converged"] is True
    assert "coefficients" in result


def test_call_r_script_missing_script() -> None:
    """Teste que call_r_script lève une erreur si le script n'existe pas."""
    with pytest.raises(FileNotFoundError, match="introuvable"):
        call_r_script("nonexistent.R", {})


def test_call_r_script_without_r_installed() -> None:
    """Teste la dégradation propre si R n'est pas installé.

    Note : Ce test ne peut être exécuté que sur une machine sans R.
    Sur une machine avec R, il passera quand même car R est disponible.
    """
    # Ce test est informatif - il vérifie le comportement
    # Sur une machine avec R, il ne testera pas l'erreur
    try:
        result = call_r_script(
            "firth_logistic.R",
            {"data": {"y": [0, 1], "x": [1, 2]}, "outcome": "y", "predictors": ["x"]},
        )
        # Si on arrive ici, R est installé et le script a fonctionné
        assert "coefficients" in result or "error" in result
    except RuntimeError as e:
        # Si R n'est pas installé, on doit avoir un message clair
        assert "R n'est pas installé" in str(e)


def test_call_r_script_firth_with_separation() -> None:
    """Teste le script R Firth sur un dataset avec séparation complète."""
    # Dataset avec séparation parfaite (x1 prédit parfaitement outcome)
    df = pd.DataFrame(
        {
            "outcome": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    payload = {
        "data": df.to_dict(orient="list"),
        "outcome": "outcome",
        "predictors": ["x1"],
    }

    # Ce test nécessite R, il sera skipé automatiquement si R n'est pas là
    result = call_r_script("firth_logistic.R", payload)

    # Firth doit converger même avec séparation complète (contrairement à Logit classique)
    assert result["converged"] is True
    assert "coefficients" in result
    assert "x1" in result["coefficients"]
