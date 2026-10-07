"""Tests du système de dispatch par registre pour la génération de code.

Valide que :
- Le registre fonctionne correctement (register, get, list)
- La signature publique generate_reproducible_code n'a pas changé (T17)
- Les méthodes V1 sont automatiquement enregistrées
- Les erreurs sont claires en français
"""

from __future__ import annotations

from typing import Any

import pytest

from app.errors import StatisticalIDEError
from app.results import codegen
from app.results.codegen import (
    CodegenRegistrationError,
    generate_reproducible_code,
    list_codegen_methods,
    register_codegen,
)

# ============================================================================
# Tests du registre
# ============================================================================


def test_register_codegen_success() -> None:
    """Un générateur peut être enregistré avec un method_id unique."""
    # Nettoyage du registre pour le test (état isolé)
    method_id = "test_method_unique_xyz"
    assert method_id not in codegen._CODEGEN_REGISTRY

    def dummy_gen(params: dict[str, Any]) -> str:
        return "# dummy"

    register_codegen(method_id, dummy_gen)
    assert method_id in codegen._CODEGEN_REGISTRY
    # Nettoyage
    del codegen._CODEGEN_REGISTRY[method_id]


def test_register_codegen_duplicate_raises() -> None:
    """Enregistrer deux fois le même method_id lève CodegenRegistrationError."""
    method_id = "test_duplicate_abc"

    def gen1(params: dict[str, Any]) -> str:
        return "# gen1"

    def gen2(params: dict[str, Any]) -> str:
        return "# gen2"

    # Nettoyage préalable
    codegen._CODEGEN_REGISTRY.pop(method_id, None)

    register_codegen(method_id, gen1)
    with pytest.raises(CodegenRegistrationError) as exc_info:
        register_codegen(method_id, gen2)
    assert "déjà enregistré" in str(exc_info.value)
    # Nettoyage
    del codegen._CODEGEN_REGISTRY[method_id]


def test_list_codegen_methods_returns_sorted() -> None:
    """list_codegen_methods retourne une liste triée des method_id."""
    methods = list_codegen_methods()
    assert isinstance(methods, list)
    assert methods == sorted(methods)
    # Les méthodes V1 doivent être présentes
    assert "linear_regression" in methods
    assert "logistic_regression" in methods
    assert "chi_square" in methods
    assert "t_test" in methods
    assert "anova" in methods


# ============================================================================
# Tests de la fonction publique (signature inchangée, T17)
# ============================================================================


def test_generate_reproducible_code_unknown_method_raises() -> None:
    """Une méthode inconnue lève KeyError avec message clair en français."""
    with pytest.raises(KeyError) as exc_info:
        generate_reproducible_code("methode_inexistante", {})
    assert "non enregistrée" in str(exc_info.value)


def test_generate_reproducible_code_linear_regression() -> None:
    """La régression linéaire produit un script exécutable."""
    params = {
        "dataset_path": "/tmp/data.csv",
        "outcome": "y",
        "predictors": ["x1", "x2"],
    }
    code = generate_reproducible_code("linear_regression", params)
    assert "import pandas as pd" in code
    assert "import statsmodels.api as sm" in code
    assert "sm.OLS" in code
    assert "/tmp/data.csv" in code
    assert '"y"' in code


def test_generate_reproducible_code_missing_param_raises() -> None:
    """Un paramètre obligatoire manquant lève ValueError."""
    params = {"dataset_path": "/tmp/data.csv", "outcome": "y"}
    # predictors manque
    with pytest.raises(ValueError) as exc_info:
        generate_reproducible_code("linear_regression", params)
    assert "Paramètres manquants" in str(exc_info.value)
    assert "predictors" in str(exc_info.value)


def test_generate_reproducible_code_chi_square() -> None:
    """Le chi-carré produit un script avec scipy.stats."""
    params = {
        "dataset_path": "/tmp/data.csv",
        "var1": "A",
        "var2": "B",
    }
    code = generate_reproducible_code("chi_square", params)
    assert "chi2_contingency" in code
    assert "pd.crosstab" in code


def test_generate_reproducible_code_t_test() -> None:
    """Le test t produit un script avec ttest_ind."""
    params = {
        "dataset_path": "/tmp/data.csv",
        "var1": "score_avant",
        "var2": "score_apres",
    }
    code = generate_reproducible_code("t_test", params)
    assert "ttest_ind" in code
    assert "score_avant" in code


def test_generate_reproducible_code_anova() -> None:
    """L'ANOVA produit un script avec f_oneway."""
    params = {
        "dataset_path": "/tmp/data.csv",
        "outcome": "score",
        "groups": "traitement",
    }
    code = generate_reproducible_code("anova", params)
    assert "f_oneway" in code
    assert "traitement" in code


def test_generate_reproducible_code_logistic_with_reference() -> None:
    """La logistique avec référence inclut le reorder."""
    params = {
        "dataset_path": "/tmp/data.csv",
        "outcome": "y",
        "predictors": ["x1"],
        "reference_category": "A",
    }
    code = generate_reproducible_code("logistic_regression", params)
    assert "reorder_categories" in code
    assert "Catégorie de référence" in code


# ============================================================================
# Tests de rétrocompatibilité V1
# ============================================================================


def test_v1_methods_auto_registered() -> None:
    """Les 5 méthodes V1 sont automatiquement enregistrées à l'import."""
    methods = list_codegen_methods()
    expected_v1 = {
        "linear_regression",
        "logistic_regression",
        "chi_square",
        "t_test",
        "anova",
    }
    for m in expected_v1:
        assert m in methods, f"Méthode V1 '{m}' absente du registre"


def test_codegen_error_is_statistical_ide_error() -> None:
    """CodegenRegistrationError hérite bien de StatisticalIDEError."""
    assert issubclass(CodegenRegistrationError, StatisticalIDEError)
