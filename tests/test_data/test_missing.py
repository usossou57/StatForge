"""Tests pour la gestion des valeurs manquantes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.data.missing import detect_missing, impute_simple, recode_missing
from app.errors import InvalidMethodInputError


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    """Retourne un DataFrame avec des valeurs manquantes et des codes spécifiques."""
    return pd.DataFrame(
        {
            "age": [25, 30, np.nan, 40, -99],
            "score": [85.5, np.nan, 78.3, 88.7, 999],
            "groupe": ["A", "B", np.nan, "B", "A"],
        }
    )


# ============================================================================
# Tests pour detect_missing
# ============================================================================


def test_detect_missing_cas_nominal(df_with_missing: pd.DataFrame) -> None:
    """Test de détection des manquants."""
    result = detect_missing(df_with_missing)

    assert (
        result["total_missing"] == 3
    )  # 1 nan age, 1 nan score, 1 nan groupe, + les codes ne sont pas comptés ici
    assert result["pct_missing"] > 0
    assert "age" in result["by_column"]
    assert result["by_column"]["age"]["count"] == 1
    assert isinstance(result["patterns"], list)


def test_detect_missing_aucun_manquant() -> None:
    """Test de détection sur un DataFrame sans manquants."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = detect_missing(df)
    assert result["total_missing"] == 0
    assert result["pct_missing"] == 0.0
    assert result["patterns"] == []


# ============================================================================
# Tests pour recode_missing
# ============================================================================


def test_recode_missing_cas_nominal(df_with_missing: pd.DataFrame) -> None:
    """Test de recodage de valeurs spécifiques en manquants."""
    df_original = df_with_missing.copy()
    df_new, step = recode_missing(df_with_missing, "age", [-99])

    # Vérifier non-mutation
    assert df_with_missing.equals(df_original)
    # Vérifier le recodage
    assert pd.isna(df_new.loc[4, "age"])
    assert step.operation == "recode_missing"
    assert step.params["codes"] == [-99]


def test_recode_missing_plusieurs_codes(df_with_missing: pd.DataFrame) -> None:
    """Test de recodage de plusieurs codes."""
    df_new, _step = recode_missing(df_with_missing, "score", [np.nan, 999])
    # Note: np.nan dans replace peut être capricieux, on teste surtout 999
    assert pd.isna(df_new.loc[4, "score"])


def test_recode_missing_colonne_inexistante(df_with_missing: pd.DataFrame) -> None:
    """Test de recodage sur une colonne inexistante."""
    with pytest.raises(KeyError, match="n'existe pas"):
        recode_missing(df_with_missing, "inexistante", [-99])


# ============================================================================
# Tests pour impute_simple
# ============================================================================


def test_impute_simple_mean(df_with_missing: pd.DataFrame) -> None:
    """Test d'imputation par la moyenne."""
    _df_new, step = impute_simple(df_with_missing, "age", "mean")

    # La moyenne de [25, 30, 40] est 31.666...
    # Le nan et le -99 sont toujours là car on n'a pas fait de recode_missing avant
    # Mais testons sur une colonne sans codes spéciaux pour la moyenne
    df_clean = pd.DataFrame({"val": [10.0, 20.0, np.nan, 40.0]})
    df_imputed, step = impute_simple(df_clean, "val", "mean")

    assert not pd.isna(df_imputed.loc[2, "val"])
    assert df_imputed.loc[2, "val"] == pytest.approx(70.0 / 3.0)
    assert step.operation == "impute_simple"
    assert step.params["strategy"] == "mean"


def test_impute_simple_median(df_with_missing: pd.DataFrame) -> None:
    """Test d'imputation par la médiane."""
    df_clean = pd.DataFrame({"val": [10.0, 20.0, np.nan, 40.0, 50.0]})
    df_imputed, _step = impute_simple(df_clean, "val", "median")

    assert df_imputed.loc[2, "val"] == pytest.approx(30.0)


def test_impute_simple_mode(df_with_missing: pd.DataFrame) -> None:
    """Test d'imputation par le mode."""
    df_clean = pd.DataFrame({"val": ["A", "A", "B", np.nan]})
    df_imputed, _step = impute_simple(df_clean, "val", "mode")

    assert df_imputed.loc[3, "val"] == "A"


def test_impute_simple_constant(df_with_missing: pd.DataFrame) -> None:
    """Test d'imputation par une valeur constante."""
    df_clean = pd.DataFrame({"val": [1.0, 2.0, np.nan]})
    df_imputed, step = impute_simple(df_clean, "val", "constant", value=0.0)

    assert df_imputed.loc[2, "val"] == 0.0
    assert step.params["value"] == 0.0


def test_impute_simple_constant_sans_value(df_with_missing: pd.DataFrame) -> None:
    """Test d'imputation constante sans fournir de valeur."""
    df_clean = pd.DataFrame({"val": [1.0, np.nan]})
    with pytest.raises(InvalidMethodInputError, match="nécessite le paramètre 'value'"):
        impute_simple(df_clean, "val", "constant")


def test_impute_simple_strategie_invalide(df_with_missing: pd.DataFrame) -> None:
    """Test d'imputation avec une stratégie invalide."""
    df_clean = pd.DataFrame({"val": [1.0, np.nan]})
    with pytest.raises(
        InvalidMethodInputError, match="Stratégie d'imputation invalide"
    ):
        impute_simple(df_clean, "val", "knn")


def test_impute_simple_colonne_inexistante(df_with_missing: pd.DataFrame) -> None:
    """Test d'imputation sur une colonne inexistante."""
    with pytest.raises(KeyError, match="n'existe pas"):
        impute_simple(df_with_missing, "inexistante", "mean")


# ============================================================================
# Tests de non-mutation
# ============================================================================


def test_aucune_fonction_ne_mute_l_entree(df_with_missing: pd.DataFrame) -> None:
    """Vérifie qu'aucune fonction ne modifie le DataFrame d'entrée."""
    df_original = df_with_missing.copy()

    detect_missing(df_with_missing)
    assert df_with_missing.equals(df_original)

    recode_missing(df_with_missing, "age", [-99])
    assert df_with_missing.equals(df_original)

    impute_simple(df_with_missing, "age", "mean")
    assert df_with_missing.equals(df_original)
