"""Tests de reproductibilité sur 10 analyses différentes (T11.3).

Valide que chaque méthode statistique peut être :
1. Exécutée
2. Sauvegardée en JSON
3. Rejouée via execute_tool
4. Produit des résultats identiques (à 1e-10 près)
"""

import numpy as np
import pandas as pd
import pytest

from app.agent.tools import execute_tool
from app.reproducibility.project import (
    build_project_from_analysis,
    load_project,
    replay_analysis,
    save_analysis,
)

# ──────────────────────────────────────────────────────────────────────
# Tests unitaires par méthode
# ──────────────────────────────────────────────────────────────────────


class TestChiSquareReplay:
    """Test de reproductibilité pour le test du chi-carré."""

    def test_chi_square_replay(self, sample_dataset, temp_project_dir):
        """Valide que chi_square peut être sauvegardé et rejoué."""
        # Exécution originale
        result = execute_tool(
            name="run_chi_square",
            arguments={"var1": "gender", "var2": "education"},
            df=sample_dataset,
        )

        # Construction et sauvegarde du projet
        project = build_project_from_analysis(
            result=result,
            research_question="Y a-t-il une association entre genre et éducation ?",
            dataset_name="test.csv",
            dataset_rows=len(sample_dataset),
            dataset_columns=len(sample_dataset.columns),
            project_id="test_chi_square",
        )
        project_path = save_analysis(project, output_dir=temp_project_dir)

        # Chargement et rejeu
        loaded_project = load_project(project_path)
        replayed_result = replay_analysis(loaded_project, sample_dataset)

        # Vérification d'identité
        assert result.estimates["chi2"] == pytest.approx(
            replayed_result.estimates["chi2"], abs=1e-10
        )
        assert result.estimates["dof"] == replayed_result.estimates["dof"]
        assert result.p_values["chi2"] == pytest.approx(
            replayed_result.p_values["chi2"], abs=1e-10
        )


class TestTTestReplay:
    """Test de reproductibilité pour le t-test."""

    def test_t_test_replay(self, sample_dataset, temp_project_dir):
        """Valide que t_test peut être sauvegardé et rejoué."""
        result = execute_tool(
            name="run_t_test",
            arguments={"outcome": "income", "group": "gender"},
            df=sample_dataset,
        )

        project = build_project_from_analysis(
            result=result,
            research_question="Le revenu diffère-t-il selon le genre ?",
            dataset_name="test.csv",
            dataset_rows=len(sample_dataset),
            dataset_columns=len(sample_dataset.columns),
            project_id="test_t_test",
        )
        project_path = save_analysis(project, output_dir=temp_project_dir)

        loaded_project = load_project(project_path)
        replayed_result = replay_analysis(loaded_project, sample_dataset)

        # Vérification
        assert "t_statistic" in result.estimates
        assert result.estimates["t_statistic"] == pytest.approx(
            replayed_result.estimates["t_statistic"], abs=1e-10
        )


class TestANOVAReplay:
    """Test de reproductibilité pour l'ANOVA."""

    def test_anova_replay(self, sample_dataset, temp_project_dir):
        """Valide que anova peut être sauvegardé et rejoué."""
        result = execute_tool(
            name="run_anova",
            arguments={"outcome": "income", "group": "education"},
            df=sample_dataset,
        )

        project = build_project_from_analysis(
            result=result,
            research_question="Le revenu diffère-t-il selon le niveau d'éducation ?",
            dataset_name="test.csv",
            dataset_rows=len(sample_dataset),
            dataset_columns=len(sample_dataset.columns),
            project_id="test_anova",
        )
        project_path = save_analysis(project, output_dir=temp_project_dir)

        loaded_project = load_project(project_path)
        replayed_result = replay_analysis(loaded_project, sample_dataset)

        assert "f_statistic" in result.estimates
        assert result.estimates["f_statistic"] == pytest.approx(
            replayed_result.estimates["f_statistic"], abs=1e-10
        )


class TestLinearRegressionReplay:
    """Test de reproductibilité pour la régression linéaire."""

    def test_linear_regression_replay(self, sample_dataset, temp_project_dir):
        """Valide que linear_regression peut être sauvegardé et rejoué."""
        result = execute_tool(
            name="run_linear_regression",
            arguments={"outcome": "income", "predictors": ["age", "score"]},
            df=sample_dataset,
        )

        project = build_project_from_analysis(
            result=result,
            research_question="Quel est l'effet de l'âge et du score sur le revenu ?",
            dataset_name="test.csv",
            dataset_rows=len(sample_dataset),
            dataset_columns=len(sample_dataset.columns),
            project_id="test_linear_regression",
        )
        project_path = save_analysis(project, output_dir=temp_project_dir)

        loaded_project = load_project(project_path)
        replayed_result = replay_analysis(loaded_project, sample_dataset)

        # Vérification des coefficients
        for var_name in result.estimates.keys():
            assert result.estimates[var_name] == pytest.approx(
                replayed_result.estimates[var_name], abs=1e-10
            )


class TestLogisticRegressionReplay:
    """Test de reproductibilité pour la régression logistique."""

    def test_logistic_regression_replay(self, sample_dataset, temp_project_dir):
        """Valide que logistic_regression peut être sauvegardé et rejoué."""
        # Créer une variable binaire pour le test
        df = sample_dataset.copy()
        df["high_income"] = (df["income"] > df["income"].median()).astype(int)

        result = execute_tool(
            name="run_logistic_regression",
            arguments={"outcome": "high_income", "predictors": ["age", "score"]},
            df=df,
        )

        project = build_project_from_analysis(
            result=result,
            research_question="Quels facteurs prédisent un revenu élevé ?",
            dataset_name="test.csv",
            dataset_rows=len(df),
            dataset_columns=len(df.columns),
            project_id="test_logistic_regression",
        )
        project_path = save_analysis(project, output_dir=temp_project_dir)

        loaded_project = load_project(project_path)
        replayed_result = replay_analysis(loaded_project, df)

        for var_name in result.estimates.keys():
            assert result.estimates[var_name] == pytest.approx(
                replayed_result.estimates[var_name], abs=1e-10
            )


class TestMultinomialLogisticRegressionReplay:
    """Test de reproductibilité pour la régression logistique multinomiale."""

    def test_multinomial_logistic_regression_replay(
        self, sample_dataset, temp_project_dir
    ):
        """Valide que multinomial_logistic_regression peut être sauvegardé et rejoué."""
        result = execute_tool(
            name="run_multinomial_logistic_regression",
            arguments={"outcome": "satisfaction", "predictors": ["age", "income"]},
            df=sample_dataset,
        )

        project = build_project_from_analysis(
            result=result,
            research_question="Quels facteurs influencent la satisfaction ?",
            dataset_name="test.csv",
            dataset_rows=len(sample_dataset),
            dataset_columns=len(sample_dataset.columns),
            project_id="test_multinomial_logistic",
        )
        project_path = save_analysis(project, output_dir=temp_project_dir)

        loaded_project = load_project(project_path)
        replayed_result = replay_analysis(loaded_project, sample_dataset)

        # Vérification d'au moins un coefficient
        assert len(result.estimates) > 0
        first_var = list(result.estimates.keys())[0]
        assert result.estimates[first_var] == pytest.approx(
            replayed_result.estimates[first_var], abs=1e-10
        )


# ──────────────────────────────────────────────────────────────────────
# Tests d'intégration : cycle complet save → load → replay
# ──────────────────────────────────────────────────────────────────────


class TestFullReproducibilityCycle:
    """Tests d'intégration sur le cycle complet de reproductibilité."""

    @pytest.mark.parametrize(
        "method,arguments",
        [
            ("run_chi_square", {"var1": "gender", "var2": "education"}),
            ("run_t_test", {"outcome": "income", "group": "gender"}),
            ("run_anova", {"outcome": "income", "group": "education"}),
            (
                "run_linear_regression",
                {"outcome": "income", "predictors": ["age", "score"]},
            ),
        ],
    )
    def test_save_load_replay_cycle(
        self, sample_dataset, temp_project_dir, method, arguments
    ):
        """Valide le cycle complet : exécution → sauvegarde → chargement → rejeu."""
        # Exécution
        result = execute_tool(name=method, arguments=arguments, df=sample_dataset)

        # Sauvegarde
        project = build_project_from_analysis(
            result=result,
            research_question=f"Test de {method}",
            dataset_name="test.csv",
            dataset_rows=len(sample_dataset),
            dataset_columns=len(sample_dataset.columns),
            project_id=f"test_cycle_{method}",
        )
        project_path = save_analysis(project, output_dir=temp_project_dir)

        # Chargement
        loaded_project = load_project(project_path)
        assert loaded_project.project_id == project.project_id
        assert loaded_project.method == result.method

        # Rejeu
        replayed_result = replay_analysis(loaded_project, sample_dataset)

        # Vérification que les résultats sont identiques
        assert replayed_result.method == result.method
        assert len(replayed_result.estimates) == len(result.estimates)

        for key in result.estimates.keys():
            assert result.estimates[key] == pytest.approx(
                replayed_result.estimates[key], abs=1e-10
            )


# ──────────────────────────────────────────────────────────────────────
# Tests de robustesse
# ──────────────────────────────────────────────────────────────────────


class TestReproducibilityRobustness:
    """Tests de robustesse pour la reproductibilité."""

    def test_replay_with_different_seed(self, temp_project_dir):
        """Valide que le rejeu est déterministe même avec des seeds différents."""
        np.random.seed(42)
        df1 = pd.DataFrame(
            {
                "x": np.random.randn(50),
                "y": np.random.randn(50),
            }
        )

        np.random.seed(123)
        df2 = pd.DataFrame(
            {
                "x": np.random.randn(50),
                "y": np.random.randn(50),
            }
        )

        # Les deux datasets sont différents mais ont la même structure
        result1 = execute_tool(
            name="run_linear_regression",
            arguments={"outcome": "y", "predictors": ["x"]},
            df=df1,
        )

        project = build_project_from_analysis(
            result=result1,
            research_question="Test de robustesse",
            dataset_name="test.csv",
            dataset_rows=len(df1),
            dataset_columns=len(df1.columns),
            project_id="test_robustness",
        )
        project_path = save_analysis(project, output_dir=temp_project_dir)

        loaded_project = load_project(project_path)

        # Rejeu sur df1 (même données) → doit être identique
        replayed_result = replay_analysis(loaded_project, df1)
        assert result1.estimates["x"] == pytest.approx(
            replayed_result.estimates["x"], abs=1e-10
        )

        # Rejeu sur df2 (données différentes) → doit être différent
        replayed_result_df2 = replay_analysis(loaded_project, df2)
        assert result1.estimates["x"] != pytest.approx(
            replayed_result_df2.estimates["x"], abs=1e-6
        )
