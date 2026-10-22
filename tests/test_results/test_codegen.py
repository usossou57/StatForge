"""Tests pour la génération de code reproductible."""

from pathlib import Path

import pandas as pd
import pytest

from app.results.codegen import generate_reproducible_code


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Crée un dataset de test temporaire."""
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "x1": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            "x2": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
            "group": ["A", "A", "B", "B", "A", "A", "B", "B", "A", "B"],
        }
    )
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return path


def test_generate_linear_regression_code(sample_dataset: Path) -> None:
    """Teste la génération de code pour une régression linéaire."""
    params = {
        "dataset_path": str(sample_dataset),
        "outcome": "y",
        "predictors": ["x1", "x2"],
    }
    code = generate_reproducible_code("linear_regression", params)

    # Vérifie que le code contient les éléments attendus
    assert "import pandas as pd" in code
    assert "import statsmodels.api as sm" in code
    assert str(sample_dataset) in code
    assert "sm.OLS" in code

    # Exécute le code généré (seul endroit où exec() est acceptable)
    exec(compile(code, "<generated>", "exec"), {"__name__": "__main__"})


def test_generate_logistic_regression_code(sample_dataset: Path) -> None:
    """Teste la génération de code pour une régression logistique."""
    # Crée un dataset binaire pour la logistique
    df = pd.DataFrame(
        {
            "outcome": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "x2": [0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        }
    )
    path = sample_dataset.parent / "logistic_data.csv"
    df.to_csv(path, index=False)

    params = {
        "dataset_path": str(path),
        "outcome": "outcome",
        "predictors": ["x1", "x2"],
    }
    code = generate_reproducible_code("logistic_regression", params)

    assert "import statsmodels.formula.api as smf" in code
    assert "smf.logit" in code

    # Exécute le code généré
    exec(compile(code, "<generated>", "exec"), {"__name__": "__main__"})


def test_generate_chi_square_code(sample_dataset: Path) -> None:
    """Teste la génération de code pour un test du chi-carré."""
    params = {
        "dataset_path": str(sample_dataset),
        "var1": "x2",
        "var2": "group",
    }
    code = generate_reproducible_code("chi_square", params)

    assert "from scipy import stats" in code
    assert "stats.chi2_contingency" in code
    assert "pd.crosstab" in code

    # Exécute le code généré
    exec(compile(code, "<generated>", "exec"), {"__name__": "__main__"})


def test_generate_t_test_code(sample_dataset: Path) -> None:
    """Teste la génération de code pour un test t."""
    params = {
        "dataset_path": str(sample_dataset),
        "var1": "x1",
        "var2": "y",
    }
    code = generate_reproducible_code("t_test", params)

    assert "stats.ttest_ind" in code

    # Exécute le code généré
    exec(compile(code, "<generated>", "exec"), {"__name__": "__main__"})


def test_generate_anova_code(sample_dataset: Path) -> None:
    """Teste la génération de code pour une ANOVA."""
    params = {
        "dataset_path": str(sample_dataset),
        "outcome": "y",
        "groups": "group",
    }
    code = generate_reproducible_code("anova", params)

    assert "stats.f_oneway" in code

    # Exécute le code généré
    exec(compile(code, "<generated>", "exec"), {"__name__": "__main__"})


def test_unsupported_method_raises_error() -> None:
    """Teste qu'une méthode non supportée lève une KeyError."""
    params = {"dataset_path": "dummy.csv"}

    with pytest.raises(KeyError, match="non enregistrée"):
        generate_reproducible_code("unknown_method", params)


def test_missing_required_params_raises_error() -> None:
    """Teste que des paramètres manquants lèvent une ValueError."""
    params = {"dataset_path": "dummy.csv"}  # Manque 'outcome' et 'predictors'

    with pytest.raises(ValueError, match="Paramètres manquants"):
        generate_reproducible_code("linear_regression", params)
