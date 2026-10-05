"""Tests pour la génération de rapports et graphiques."""

from datetime import datetime, timezone

import matplotlib
import matplotlib.pyplot as plt
import pytest

from app.diagnostics.models import DiagnosticFinding, Severity
from app.reports.generator import generate_plot, generate_report
from app.results.models import AnalysisResult

# Utiliser un backend non-interactif pour les tests
matplotlib.use("Agg")


@pytest.fixture
def sample_analysis_result() -> AnalysisResult:
    """Crée un AnalysisResult de test."""
    return AnalysisResult(
        method="linear_regression",
        dataset_id="test_dataset_001",
        variables={
            "y": {"role": "outcome", "dtype": "numeric"},
            "x1": {"role": "predictor", "dtype": "numeric"},
            "x2": {"role": "predictor", "dtype": "numeric"},
        },
        sample_size=100,
        preprocessing=["Suppression des valeurs manquantes"],
        estimates={"x1": 2.5, "x2": -1.3, "const": 0.5},
        standard_errors={"x1": 0.3, "x2": 0.4, "const": 0.2},
        confidence_intervals={
            "x1": (1.9, 3.1),
            "x2": (-2.1, -0.5),
            "const": (0.1, 0.9),
        },
        p_values={"x1": 0.001, "x2": 0.01, "const": 0.05},
        model_metrics={"r_squared": 0.75, "adj_r_squared": 0.73},
        diagnostics=[
            DiagnosticFinding(
                code="HIGH_VIF",
                severity=Severity.PROBABLE_ISSUE,
                message="Multicolinéarité détectée pour x1 (VIF = 6.2)",
                details={"variable": "x1", "vif": 6.2},
            )
        ],
        warnings=["Quelques valeurs manquantes supprimées"],
        assumptions_checked=["Normalité des résidus", "Homoscédasticité"],
        generated_code="import pandas as pd\n# Code reproductible ici",
        execution_log=["Analyse démarrée", "Analyse terminée"],
        software_versions={"python": "3.11.0", "pandas": "2.0.0"},
        created_at=datetime(2026, 9, 22, 10, 0, 0, tzinfo=timezone.utc),
    )


def test_generate_html_report(sample_analysis_result: AnalysisResult) -> None:
    """Teste la génération d'un rapport HTML."""
    report = generate_report(sample_analysis_result, output_format="html")

    # Vérifie que le rapport contient les éléments essentiels
    assert "<!DOCTYPE html>" in report
    assert "linear_regression" in report
    assert "test_dataset_001" in report
    assert "100" in report  # sample_size
    assert "x1" in report
    assert "x2" in report
    assert "HIGH_VIF" in report
    assert "Code reproductible" in report
    assert "python" in report


def test_generate_markdown_report(sample_analysis_result: AnalysisResult) -> None:
    """Teste la génération d'un rapport Markdown."""
    report = generate_report(sample_analysis_result, output_format="markdown")

    # Vérifie que le rapport contient les éléments essentiels
    assert "# Rapport d'analyse" in report
    assert "linear_regression" in report
    assert "test_dataset_001" in report
    assert "## Variables" in report
    assert "## Coefficients" in report
    assert "x1" in report
    assert "## Diagnostics" in report
    assert "HIGH_VIF" in report
    assert "## Code reproductible" in report
    assert "```python" in report


def test_generate_report_invalid_format(sample_analysis_result: AnalysisResult) -> None:
    """Teste qu'un format invalide lève une ValueError."""
    with pytest.raises(ValueError, match="non supporté"):
        generate_report(sample_analysis_result, output_format="pdf")


def test_generate_coefficients_plot(sample_analysis_result: AnalysisResult) -> None:
    """Teste la génération d'un forest plot des coefficients."""
    fig = generate_plot(sample_analysis_result, kind="coefficients")

    # Vérifie que c'est bien une figure matplotlib
    assert isinstance(fig, matplotlib.figure.Figure)

    # Vérifie que la figure contient un axe avec des données
    assert len(fig.axes) > 0
    ax = fig.axes[0]
    assert ax.get_title() != ""  # Titre présent
    assert ax.get_xlabel() != ""  # Label X présent

    # Ferme la figure pour éviter les fuites mémoire
    plt.close(fig)


def test_generate_plot_unsupported_kind(sample_analysis_result: AnalysisResult) -> None:
    """Teste qu'un type de graphique non supporté lève une ValueError."""
    with pytest.raises(ValueError, match="non supporté"):
        generate_plot(sample_analysis_result, kind="unknown")


def test_generate_plot_residuals_not_implemented(
    sample_analysis_result: AnalysisResult,
) -> None:
    """Teste que le graphique de résidus lève NotImplementedError."""
    with pytest.raises(NotImplementedError, match="pas encore implémenté"):
        generate_plot(sample_analysis_result, kind="residuals")


def test_generate_report_with_empty_diagnostics() -> None:
    """Teste la génération de rapport sans diagnostics."""
    result = AnalysisResult(
        method="t_test",
        dataset_id="test_002",
        variables={"y": {"role": "outcome"}, "group": {"role": "predictor"}},
        sample_size=50,
        preprocessing=[],
        estimates={"diff": 1.5},
        standard_errors={"diff": 0.3},
        confidence_intervals={"diff": (0.9, 2.1)},
        p_values={"diff": 0.001},
        model_metrics={},
        diagnostics=[],  # Pas de diagnostics
        warnings=[],
        assumptions_checked=[],
        generated_code="# Code vide",
        execution_log=[],
        software_versions={"python": "3.11.0"},
    )

    # Ne doit pas lever d'erreur
    html_report = generate_report(result, output_format="html")
    md_report = generate_report(result, output_format="markdown")

    assert "t_test" in html_report
    assert "t_test" in md_report


def test_generate_plot_with_no_coefficients() -> None:
    """Teste la génération de graphique sans coefficients."""
    result = AnalysisResult(
        method="chi_square",
        dataset_id="test_003",
        variables={"var1": {"role": "predictor"}, "var2": {"role": "predictor"}},
        sample_size=100,
        preprocessing=[],
        estimates={},  # Pas de coefficients pour chi²
        standard_errors={},
        confidence_intervals={},
        p_values={"chi2": 0.05},
        model_metrics={},
        diagnostics=[],
        warnings=[],
        assumptions_checked=[],
        generated_code="# Code chi²",
        execution_log=[],
        software_versions={},
    )

    # Ne doit pas lever d'erreur, retourne une figure avec message
    fig = generate_plot(result, kind="coefficients")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
