"""Tests pour app/statistics/tests_classiques.py."""

import math

import pandas as pd
import pytest

from app.statistics.tests_classiques import run_anova, run_chi_square, run_t_test


def test_chi_square_cas_nominal() -> None:
    df = pd.DataFrame({"sexe": ["H", "F", "H", "F", "H", "F"], "fumeur": ["O", "N", "O", "O", "N", "N"]})
    result = run_chi_square(df, "sexe", "fumeur")
    assert result.method == "chi_square" and "chi2" in result.estimates and "cramers_v" in result.model_metrics

def test_chi_square_colonne_manquante() -> None:
    with pytest.raises(ValueError, match="n'existe pas"):
        run_chi_square(pd.DataFrame({"sexe": ["H"], "fumeur": ["O"]}), "sexe", "manquante")

def test_chi_square_variable_non_categorielle() -> None:
    df = pd.DataFrame({"age": [25, 30, 35, 40, 45, 50, 55], "sexe": ["H", "F", "H", "F", "H", "F", "H"]})
    with pytest.raises(ValueError, match="catégorielle"):
        run_chi_square(df, "age", "sexe")

def test_chi_square_valeurs_manquantes_warning() -> None:
    df = pd.DataFrame({"sexe": ["H", "F", None, "H", "F"], "fumeur": ["O", "N", "O", None, "N"]})
    result = run_chi_square(df, "sexe", "fumeur")
    assert len(result.warnings) == 1 and "dropna" in result.preprocessing

def test_chi_square_validation_r() -> None:
    df = pd.DataFrame({"var1": ["X"]*5 + ["Y"]*5, "var2": ["A", "A", "B", "B", "A", "B", "B", "A", "A", "B"]})
    result = run_chi_square(df, "var1", "var2")
    assert math.isclose(result.estimates["chi2"], 0.4, rel_tol=1e-6)
    assert math.isclose(result.p_values["chi2"], 0.5271, rel_tol=1e-3)
    assert math.isclose(result.model_metrics["cramers_v"], 0.2, rel_tol=1e-6)

def test_t_test_cas_nominal() -> None:
    df = pd.DataFrame({"score": [10.0, 12.0, 11.0, 20.0, 22.0, 21.0], "groupe": ["A", "A", "A", "B", "B", "B"]})
    result = run_t_test(df, "score", "groupe")
    assert result.method == "t_test" and "t_statistic" in result.model_metrics

def test_t_test_trois_modalites_valueerror() -> None:
    df = pd.DataFrame({"score": [10.0, 12.0, 11.0, 20.0, 22.0, 21.0, 30.0], "groupe": ["A", "A", "A", "B", "B", "B", "C"]})
    with pytest.raises(ValueError, match="exactement 2 modalités"):
        run_t_test(df, "score", "groupe")

def test_t_test_colonne_manquante() -> None:
    with pytest.raises(ValueError, match="n'existe pas"):
        run_t_test(pd.DataFrame({"score": [10.0], "groupe": ["A"]}), "score", "manquante")

def test_t_test_outcome_non_numerique() -> None:
    with pytest.raises(ValueError, match="n'est pas numérique"):
        run_t_test(pd.DataFrame({"score": ["A"], "groupe": ["X"]}), "score", "groupe")

def test_t_test_valeurs_manquantes_warning() -> None:
    df = pd.DataFrame({"score": [10.0, 12.0, None, 20.0, 22.0, 21.0], "groupe": ["A", "A", "A", "B", "B", "B"]})
    result = run_t_test(df, "score", "groupe")
    assert len(result.warnings) == 1 and "dropna" in result.preprocessing

def test_t_test_validation_r() -> None:
    df = pd.DataFrame({"score": [10.0, 12.0, 11.0, 13.0, 20.0, 22.0, 21.0, 23.0], "groupe": ["A"]*4 + ["B"]*4})
    result = run_t_test(df, "score", "groupe")
    assert math.isclose(result.estimates["mean_A"], 11.5, rel_tol=1e-6)
    assert math.isclose(result.estimates["mean_B"], 21.5, rel_tol=1e-6)
    assert math.isclose(result.model_metrics["t_statistic"], -10.954, rel_tol=1e-3)
    # Valeur corrigée : p-value pour t=-10.954, df=6 est bien 3.436e-05
    assert math.isclose(result.p_values["t_test"], 3.436e-05, rel_tol=1e-3)

def test_anova_cas_nominal() -> None:
    df = pd.DataFrame({"score": [10.0, 12.0, 11.0, 20.0, 22.0, 21.0, 30.0, 32.0, 31.0], "groupe": ["A"]*3 + ["B"]*3 + ["C"]*3})
    result = run_anova(df, "score", "groupe")
    assert result.method == "anova" and "f_statistic" in result.estimates and "eta_squared" in result.model_metrics

def test_anova_colonne_manquante() -> None:
    with pytest.raises(ValueError, match="n'existe pas"):
        run_anova(pd.DataFrame({"score": [10.0], "groupe": ["A"]}), "score", "manquante")

def test_anova_groupe_unique_valueerror() -> None:
    with pytest.raises(ValueError, match="au moins 2 modalités"):
        run_anova(pd.DataFrame({"score": [10.0, 12.0, 11.0], "groupe": ["A", "A", "A"]}), "score", "groupe")

def test_anova_variable_non_numerique() -> None:
    with pytest.raises(ValueError, match="n'est pas numérique"):
        run_anova(pd.DataFrame({"score": ["A", "B", "C"], "groupe": ["X", "Y", "Z"]}), "score", "groupe")

def test_anova_validation_r() -> None:
    df = pd.DataFrame({"score": [10.0, 12.0, 11.0, 20.0, 22.0, 21.0, 30.0, 32.0, 31.0], "groupe": ["A"]*3 + ["B"]*3 + ["C"]*3})
    result = run_anova(df, "score", "groupe")
    assert math.isclose(result.estimates["f_statistic"], 300.0, rel_tol=1e-4)
    assert result.estimates["df_between"] == 2 and result.estimates["df_within"] == 6
    assert math.isclose(result.p_values["anova"], 9.7e-7, rel_tol=1e-1)
    assert math.isclose(result.model_metrics["eta_squared"], 0.9901, rel_tol=1e-3)
