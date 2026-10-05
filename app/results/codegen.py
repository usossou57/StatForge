"""Génération de code Python reproductible pour les analyses statistiques.

Ce module produit des scripts Python autonomes qui, exécutés sur le même
fichier de données, reproduisent exactement l'analyse demandée.
"""

from __future__ import annotations

import textwrap
from typing import Any

# Méthodes statistiques supportées par le moteur
_SUPPORTED_METHODS: frozenset[str] = frozenset({
    "linear_regression",
    "logistic_regression",
    "chi_square",
    "t_test",
    "anova",
})


def generate_reproducible_code(method: str, params: dict[str, Any]) -> str:
    """Retourne un script Python autonome (imports inclus) qui, exécuté
    tel quel sur le même fichier de données, reproduit l'analyse.

    Args:
        method: Nom de la méthode statistique. Doit être parmi :
            'linear_regression', 'logistic_regression', 'chi_square',
            't_test', 'anova'.
        params: Dictionnaire des paramètres nécessaires à la reproduction.
            Clés requises selon la méthode :
            - Toutes : 'dataset_path' (str)
            - linear_regression : 'outcome' (str), 'predictors' (list[str])
            - logistic_regression : 'outcome' (str), 'predictors' (list[str]),
              'reference_category' (str | None)
            - chi_square : 'var1' (str), 'var2' (str)
            - t_test : 'var1' (str), 'var2' (str), 'paired' (bool, défaut False)
            - anova : 'outcome' (str), 'groups' (str)

    Returns:
        Un script Python complet et autonome sous forme de string.

    Raises:
        ValueError: Si la méthode n'est pas supportée.
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Méthode '{method}' non supportée. "
            f"Méthodes disponibles : {', '.join(sorted(_SUPPORTED_METHODS))}."
        )

    generators = {
        "linear_regression": _gen_linear_regression,
        "logistic_regression": _gen_logistic_regression,
        "chi_square": _gen_chi_square,
        "t_test": _gen_t_test,
        "anova": _gen_anova,
    }

    return generators[method](params)


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
            print(f"Dataset chargé : {{df.shape[0]}} lignes, {{df.shape[1]}} colonnes")

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
            print(f"R² ajusté : {{model.rsquared_adj:.4f}}")
            print(f"F-statistic : {{model.fvalue:.4f}} (p = {{model.f_pvalue:.4e}})")
            print("\\nCoefficients :")
            for name, coef in model.params.items():
                pval = model.pvalues[name]
                print(f"  {{name}}: {{coef:.4f}} (p = {{pval:.4e}})")


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
        ref_line = f'    df["{outcome}"] = df["{outcome}"].cat.reorder_categories(\\n        df["{outcome}"].cat.categories.tolist()\\n    )\\n    # Catégorie de référence : "{reference_category}"\\n'

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
            print(f"Dataset chargé : {{df.shape[0]}} lignes, {{df.shape[1]}} colonnes")

            # Préparation des variables
            outcome = "{outcome}"
            predictors = {predictors!r}
        {ref_line}
            # Construction de la formule
            formula = f"{{outcome}} ~ {{' + '.join(predictors)}}"
            model = smf.logit(formula, data=df).fit(disp=0)

            # Affichage des résultats
            print("\\n=== Résultats de la régression logistique ===")
            print(f"Pseudo-R² : {{model.prsquared:.4f}}")
            print(f"Log-Likelihood : {{model.llf:.4f}}")
            print(f"AIC : {{model.aic:.4f}}")
            print("\\nCoefficients (log-odds) :")
            for name, coef in model.params.items():
                pval = model.pvalues[name]
                print(f"  {{name}}: {{coef:.4f}} (p = {{pval:.4e}})")


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
            print(f"Dataset chargé : {{df.shape[0]}} lignes, {{df.shape[1]}} colonnes")

            # Variables
            var1 = "{var1}"
            var2 = "{var2}"

            # Table de contingence
            contingency = pd.crosstab(df[var1], df[var2])
            print(f"\\nTable de contingence ({var1} x {var2}) :")
            print(contingency)

            # Test du chi-carré
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

            # Affichage des résultats
            print("\\n=== Résultats du test du chi-carré ===")
            print(f"Chi² = {{chi2:.4f}}")
            print(f"ddl = {{dof}}")
            print(f"p-value = {{p_value:.4e}}")
            print(f"Résultat : {{'Rejet de H0' if p_value < 0.05 else 'Non-rejet de H0'}} (alpha = 0.05)")


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
            print(f"Dataset chargé : {{df.shape[0]}} lignes, {{df.shape[1]}} colonnes")

            # Variables
            var1 = "{var1}"
            var2 = "{var2}"
            paired = {paired!s}

            # Extraction des échantillons
            sample1 = df[var1].dropna()
            sample2 = df[var2].dropna()

            # Statistiques descriptives
            print(f"\\n=== Statistiques descriptives ===")
            print(f"  {var1}: n={{len(sample1)}}, mean={{sample1.mean():.4f}}, std={{sample1.std():.4f}}")
            print(f"  {var2}: n={{len(sample2)}}, mean={{sample2.mean():.4f}}, std={{sample2.std():.4f}}")

            # Test t
            t_stat, p_value = stats.ttest_ind(
                sample1, sample2, equal_var=False
            )

            # Affichage des résultats
            print("\\n=== Résultats du test t de Student ===")
            print(f"t = {{t_stat:.4f}}")
            print(f"p-value = {{p_value:.4e}}")
            print(f"Type : {{'apparié' if paired else 'indépendant'}} (Welch)")
            print(f"Résultat : {{'Rejet de H0' if p_value < 0.05 else 'Non-rejet de H0'}} (alpha = 0.05)")


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
            print(f"Dataset chargé : {{df.shape[0]}} lignes, {{df.shape[1]}} colonnes")

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
                print(f"  {{label}}: n={{len(sample)}}, mean={{sample.mean():.4f}}, std={{sample.std():.4f}}")

            # ANOVA
            f_stat, p_value = stats.f_oneway(*samples)

            # Affichage des résultats
            print("\\n=== Résultats de l'ANOVA ===")
            print(f"F = {{f_stat:.4f}}")
            print(f"p-value = {{p_value:.4e}}")
            print(f"Nombre de groupes : {{len(group_labels)}}")
            print(f"Résultat : {{'Rejet de H0' if p_value < 0.05 else 'Non-rejet de H0'}} (alpha = 0.05)")


        if __name__ == "__main__":
            main()
    """)
