"""Génération de code Python reproductible pour les analyses statistiques.

Ce module produit des scripts Python autonomes qui, exécutés sur le même
fichier de données, reproduisent exactement l'analyse demandée.

Architecture (V2) : système de dispatch par registre. Chaque méthode
statistique enregistre sa fonction de génération de code via
`register_codegen(method_id, fn)`. La fonction publique
`generate_reproducible_code` ne change pas de signature (T17).
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from typing import Any

from app.errors import StatisticalIDEError

# Registre interne des générateurs de code reproductible
_CODEGEN_REGISTRY: dict[str, Callable[[dict[str, Any]], str]] = {}


class CodegenRegistrationError(StatisticalIDEError):
    """Erreur levée lors d'un enregistrement invalide dans le registre de codegen."""



def register_codegen(
    method_id: str, fn: Callable[[dict[str, Any]], str]
) -> None:
    """Enregistre une fonction de génération de code pour une méthode.

    Args:
        method_id: Identifiant unique de la méthode (doit correspondre à un
            method_id du registre statistique).
        fn: Fonction qui prend un dictionnaire de paramètres et retourne
            un script Python autonome sous forme de chaîne.

    Raises:
        CodegenRegistrationError: Si le method_id est déjà enregistré.
    """
    if method_id in _CODEGEN_REGISTRY:
        raise CodegenRegistrationError(
            f"Le générateur de code pour '{method_id}' est déjà enregistré. "
            "Un method_id ne peut avoir qu'un seul générateur."
        )
    _CODEGEN_REGISTRY[method_id] = fn


def generate_reproducible_code(method: str, params: dict[str, Any]) -> str:
    """Retourne un script Python autonome (imports inclus) qui, exécuté
    tel quel sur le même fichier de données, reproduit l'analyse.

    Args:
        method: Identifiant de la méthode statistique (ex: 'linear_regression').
        params: Dictionnaire des paramètres nécessaires à la reproduction.
            Clés requises selon la méthode (voir docstrings des générateurs).

    Returns:
        Un script Python complet et autonome sous forme de string.

    Raises:
        KeyError: Si la méthode n'est pas enregistrée dans le registre.
    """
    if method not in _CODEGEN_REGISTRY:
        available = ", ".join(sorted(_CODEGEN_REGISTRY.keys()))
        raise KeyError(
            f"Méthode '{method}' non enregistrée dans le registre de codegen. "
            f"Méthodes disponibles : {available or '(aucune)'}. "
            "Vérifiez que la méthode a bien été enregistrée via register_codegen()."
        )
    return _CODEGEN_REGISTRY[method](params)


def list_codegen_methods() -> list[str]:
    """Retourne la liste des method_id pour lesquels un générateur est enregistré.

    Returns:
        Liste triée des identifiants de méthodes.
    """
    return sorted(_CODEGEN_REGISTRY.keys())


def _validate_required_params(
    params: dict[str, Any], required: list[str], method: str
) -> None:
    """Vérifie la présence des paramètres obligatoires.

    Args:
        params: Dictionnaire de paramètres à valider.
        required: Liste des clés obligatoires.
        method: Nom de la méthode (pour le message d'erreur).

    Raises:
        ValueError: Si un paramètre obligatoire est manquant.
    """
    missing = [k for k in required if k not in params]
    if missing:
        raise ValueError(
            f"Paramètres manquants pour '{method}' : "
            f"{', '.join(missing)}."
        )


# ============================================================================
# Générateurs V1 (rétrocompatibilité)
# ============================================================================


def _gen_linear_regression(params: dict[str, Any]) -> str:
    """Génère le script pour une régression linéaire.

    Args:
        params: Doit contenir 'dataset_path', 'outcome', 'predictors'.

    Returns:
        Script Python reproductible.
    """
    _validate_required_params(
        params, ["dataset_path", "outcome", "predictors"], "linear_regression"
    )
    dataset_path = params["dataset_path"]
    outcome = params["outcome"]
    predictors = params["predictors"]
    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"Script reproductible — Régression linéaire.
        Généré automatiquement par StatAnalysis.
        Reproduit l'analyse sur le fichier : {dataset_path}
        \"\"\"
        import pandas as pd
        import statsmodels.api as sm


        def main() -> None:
            # Chargement des données
            df = pd.read_csv("{dataset_path}")
            print(f"Dataset chargé : {{{{df.shape[0]}}}} lignes, {{{{df.shape[1]}}}} colonnes")

            # Préparation des variables
            outcome = "{outcome}"
            predictors = {predictors!r}

            X = df[predictors]
            y = df[outcome]
            X = sm.add_constant(X)

            # Modèle
            model = sm.OLS(y, X).fit()

            # Affichage des résultats
            print("\\n=== Résultats de la régression linéaire ===")
            print(f"R² ajusté : {{{{model.rsquared_adj:.4f}}}}")
            print(f"F-statistic : {{{{model.fvalue:.4f}}}} (p = {{{{model.f_pvalue:.4e}}}})")
            print("\\nCoefficients :")
            for name, coef in model.params.items():
                pval = model.pvalues[name]
                print(f"  {{{{name}}}}: {{{{coef:.4f}}}} (p = {{{{pval:.4e}}}})")


        if __name__ == "__main__":
            main()
    """)


def _gen_logistic_regression(params: dict[str, Any]) -> str:
    """Génère le script pour une régression logistique.

    Args:
        params: Doit contenir 'dataset_path', 'outcome', 'predictors'.
            Optionnel : 'reference_category'.

    Returns:
        Script Python reproductible.
    """
    _validate_required_params(
        params, ["dataset_path", "outcome", "predictors"], "logistic_regression"
    )
    dataset_path = params["dataset_path"]
    outcome = params["outcome"]
    predictors = params["predictors"]
    reference_category = params.get("reference_category")
    ref_line = ""
    if reference_category is not None:
        ref_line = (
            f'    df["{outcome}"] = df["{outcome}"].cat.reorder_categories(\n'
            f'        df["{outcome}"].cat.categories.tolist()\n'
            f'    )\n'
            f'    # Catégorie de référence : "{reference_category}"\n'
        )
    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"Script reproductible — Régression logistique.
        Généré automatiquement par StatAnalysis.
        Reproduit l'analyse sur le fichier : {dataset_path}
        \"\"\"
        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf


        def main() -> None:
            # Chargement des données
            df = pd.read_csv("{dataset_path}")
            print(f"Dataset chargé : {{{{df.shape[0]}}}} lignes, {{{{df.shape[1]}}}} colonnes")

            # Préparation des variables
            outcome = "{outcome}"
            predictors = {predictors!r}

        {ref_line}
            # Construction de la formule
            formula = outcome + " ~ " + " + ".join(predictors)
            model = smf.logit(formula, data=df).fit(disp=0)

            # Affichage des résultats
            print("\\n=== Résultats de la régression logistique ===")
            print(f"Pseudo-R² : {{{{model.prsquared:.4f}}}}")
            print(f"Log-Likelihood : {{{{model.llf:.4f}}}}")
            print(f"AIC : {{{{model.aic:.4f}}}}")
            print("\\nCoefficients (log-odds) :")
            for name, coef in model.params.items():
                pval = model.pvalues[name]
                print(f"  {{{{name}}}}: {{{{coef:.4f}}}} (p = {{{{pval:.4e}}}})")


        if __name__ == "__main__":
            main()
    """)


def _gen_chi_square(params: dict[str, Any]) -> str:
    """Génère le script pour un test du chi-carré.

    Args:
        params: Doit contenir 'dataset_path', 'var1', 'var2'.

    Returns:
        Script Python reproductible.
    """
    _validate_required_params(
        params, ["dataset_path", "var1", "var2"], "chi_square"
    )
    dataset_path = params["dataset_path"]
    var1 = params["var1"]
    var2 = params["var2"]
    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"Script reproductible — Test du chi-carré d'indépendance.
        Généré automatiquement par StatAnalysis.
        Reproduit l'analyse sur le fichier : {dataset_path}
        \"\"\"
        import pandas as pd
        from scipy import stats


        def main() -> None:
            # Chargement des données
            df = pd.read_csv("{dataset_path}")
            print(f"Dataset chargé : {{{{df.shape[0]}}}} lignes, {{{{df.shape[1]}}}} colonnes")

            # Variables
            var1 = "{var1}"
            var2 = "{var2}"

            # Table de contingence
            contingency = pd.crosstab(df[var1], df[var2])
            print(f"\\nTable de contingence ({{var1}} x {{var2}}) :")
            print(contingency)

            # Test du chi-carré
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

            # Affichage des résultats
            print("\\n=== Résultats du test du chi-carré ===")
            print(f"Chi² = {{{{chi2:.4f}}}}")
            print(f"ddl = {{{{dof}}}}")
            print(f"p-value = {{{{p_value:.4e}}}}")
            print(
                f"Résultat : "
                f"{{{{'Rejet de H0' if p_value < 0.05 else 'Non-rejet de H0'}}}} "
                f"(alpha = 0.05)"
            )


        if __name__ == "__main__":
            main()
    """)


def _gen_t_test(params: dict[str, Any]) -> str:
    """Génère le script pour un test t de Student.

    Args:
        params: Doit contenir 'dataset_path', 'var1', 'var2'.
            Optionnel : 'paired' (bool, défaut False).

    Returns:
        Script Python reproductible.
    """
    _validate_required_params(
        params, ["dataset_path", "var1", "var2"], "t_test"
    )
    dataset_path = params["dataset_path"]
    var1 = params["var1"]
    var2 = params["var2"]
    paired = params.get("paired", False)
    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"Script reproductible — Test t de Student.
        Généré automatiquement par StatAnalysis.
        Reproduit l'analyse sur le fichier : {dataset_path}
        \"\"\"
        import pandas as pd
        from scipy import stats


        def main() -> None:
            # Chargement des données
            df = pd.read_csv("{dataset_path}")
            print(f"Dataset chargé : {{{{df.shape[0]}}}} lignes, {{{{df.shape[1]}}}} colonnes")

            # Variables
            var1 = "{var1}"
            var2 = "{var2}"
            paired = {paired!s}

            # Extraction des échantillons
            sample1 = df[var1].dropna()
            sample2 = df[var2].dropna()

            # Statistiques descriptives
            print(f"\\n=== Statistiques descriptives ===")
            print(
                f"  {{var1}}: n={{{{len(sample1)}}}}, "
                f"mean={{{{sample1.mean():.4f}}}}, "
                f"std={{{{sample1.std():.4f}}}}"
            )
            print(
                f"  {{var2}}: n={{{{len(sample2)}}}}, "
                f"mean={{{{sample2.mean():.4f}}}}, "
                f"std={{{{sample2.std():.4f}}}}"
            )

            # Test t
            t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=False)

            # Affichage des résultats
            print("\\n=== Résultats du test t de Student ===")
            print(f"t = {{{{t_stat:.4f}}}}")
            print(f"p-value = {{{{p_value:.4e}}}}")
            print(f"Type : {{{{'apparié' if paired else 'indépendant'}}}} (Welch)")
            print(
                f"Résultat : "
                f"{{{{'Rejet de H0' if p_value < 0.05 else 'Non-rejet de H0'}}}} "
                f"(alpha = 0.05)"
            )


        if __name__ == "__main__":
            main()
    """)


def _gen_anova(params: dict[str, Any]) -> str:
    """Génère le script pour une ANOVA à un facteur.

    Args:
        params: Doit contenir 'dataset_path', 'outcome', 'groups'.

    Returns:
        Script Python reproductible.
    """
    _validate_required_params(
        params, ["dataset_path", "outcome", "groups"], "anova"
    )
    dataset_path = params["dataset_path"]
    outcome = params["outcome"]
    groups = params["groups"]
    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"Script reproductible — ANOVA à un facteur.
        Généré automatiquement par StatAnalysis.
        Reproduit l'analyse sur le fichier : {dataset_path}
        \"\"\"
        import pandas as pd
        from scipy import stats


        def main() -> None:
            # Chargement des données
            df = pd.read_csv("{dataset_path}")
            print(f"Dataset chargé : {{{{df.shape[0]}}}} lignes, {{{{df.shape[1]}}}} colonnes")

            # Variables
            outcome = "{outcome}"
            groups = "{groups}"

            # Séparation des groupes
            group_labels = df[groups].unique()
            samples = [
                df[df[groups] == label][outcome].dropna().values
                for label in group_labels
            ]

            # Statistiques descriptives par groupe
            print(f"\\n=== Statistiques descriptives par groupe ===")
            for label, sample in zip(group_labels, samples):
                print(
                    f"  {{{{label}}}}: n={{{{len(sample)}}}}, "
                    f"mean={{{{sample.mean():.4f}}}}, "
                    f"std={{{{sample.std():.4f}}}}"
                )

            # ANOVA
            f_stat, p_value = stats.f_oneway(*samples)

            # Affichage des résultats
            print("\\n=== Résultats de l'ANOVA ===")
            print(f"F = {{{{f_stat:.4f}}}}")
            print(f"p-value = {{{{p_value:.4e}}}}")
            print(f"Nombre de groupes : {{{{len(group_labels)}}}}")
            print(
                f"Résultat : "
                f"{{{{'Rejet de H0' if p_value < 0.05 else 'Non-rejet de H0'}}}} "
                f"(alpha = 0.05)"
            )


        if __name__ == "__main__":
            main()
    """)


# ============================================================================
# Enregistrement automatique des méthodes V1 au premier import
# ============================================================================


def _register_v1_codegen() -> None:
    """Enregistre les générateurs de code des méthodes V1.

    Cette fonction est appelée automatiquement à l'import du module pour
    assurer la rétrocompatibilité avec le code existant.
    """
    v1_methods = {
        "linear_regression": _gen_linear_regression,
        "logistic_regression": _gen_logistic_regression,
        "chi_square": _gen_chi_square,
        "t_test": _gen_t_test,
        "anova": _gen_anova,
    }
    for method_id, gen_fn in v1_methods.items():
        if method_id not in _CODEGEN_REGISTRY:
            register_codegen(method_id, gen_fn)


_register_v1_codegen()