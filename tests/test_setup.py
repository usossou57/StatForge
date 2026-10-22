"""Test de validation de la structure de base du projet.

Ce module contient un test trivial qui vérifie que l'import du package app
fonctionne correctement. Il sert uniquement à valider que la CI est
correctement câblée et que la structure de base du projet est fonctionnelle.
"""


def test_import_app() -> None:
    """Vérifie que le package app peut être importé sans erreur.

    Ce test minimal valide que :
    - Le répertoire app/ existe
    - Le fichier app/__init__.py est présent
    - Python peut charger le module sans erreur d'import

    Raises:
        ImportError: Si le module app ne peut pas être importé.
    """
    import app

    assert app is not None
