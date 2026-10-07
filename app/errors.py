"""Erreurs métier personnalisées pour l'IDE Statistique."""

from __future__ import annotations


class StatisticalIDEError(Exception):
    """Erreur de base pour toutes les erreurs spécifiques de l'application.

    Toutes les erreurs métier doivent hériter de cette classe pour être
    facilement capturées et affichées avec un message clair à l'utilisateur.
    """

    def __init__(self, message: str = "Une erreur interne est survenue.") -> None:
        self.message = message
        super().__init__(self.message)


class DependencyMissingError(StatisticalIDEError):
    """Levée lorsqu'une dépendance optionnelle (extra) n'est pas installée.

    Le message doit toujours inclure la commande exacte pour installer la
    dépendance manquante.
    """

    def __init__(self, package: str, extra: str | None = None) -> None:
        if extra:
            cmd = f"uv add --optional {extra} {package}"
        else:
            cmd = f"uv add {package}"
            
        message = (
            f"La dépendance '{package}' est requise pour cette fonctionnalité "
            f"mais n'est pas installée.\n"
            f"Veuillez l'installer avec la commande suivante :\n  {cmd}"
        )
        super().__init__(message)
        self.package = package
        self.extra = extra


class InvalidMethodInputError(StatisticalIDEError):
    """Levée lorsque les données fournies à une méthode statistique sont invalides.

    Exemples : types de variables incompatibles, taille d'échantillon insuffisante,
    données constantes, etc.
    """

    def __init__(self, message: str = "Les données fournies ne sont pas valides pour cette méthode.") -> None:
        super().__init__(message)
