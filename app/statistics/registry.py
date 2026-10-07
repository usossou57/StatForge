"""Registre central des méthodes statistiques (ADR-003).

Ce module permet d'enregistrer, découvrir et récupérer les méthodes
statistiques disponibles dans l'application. Il remplace l'ancien
système de Protocol monolithique pour permettre une architecture
modulaire et extensible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from app.errors import StatisticalIDEError


@dataclass(frozen=True)
class MethodSpec:
    """Fiche d'identité complète d'une méthode statistique.

    Attributs:
        method_id: Identifiant unique et stable de la méthode (ex: 'chi_square').
        label_fr: Libellé affichable en français pour l'UI.
        family: Famille statistique (ex: 'inférence', 'régression', 'survie').
        engine: Moteur d'exécution (Python, R, ou les deux).
        outcome_types: Types de variables acceptés pour la variable cible.
        predictor_types: Types de variables acceptés pour les prédicteurs.
        min_sample_size: Taille d'échantillon minimale recommandée.
        assumptions: Liste des hypothèses statistiques à vérifier.
        reference_impl: Référence de l'implémentation (ex: 'scipy.stats.chi2_contingency').
        r_reference: Référence du package R équivalent (ex: 'stats::chisq.test'), si applicable.
        phase: Numéro de la phase d'implémentation de la méthode.
        extra: Nom de l'extra optionnel requis (ex: 'survival'), None si aucune.
        default_params: Paramètres par défaut de la méthode.
    """

    method_id: str
    label_fr: str
    family: str
    engine: str  # 'python', 'r', ou 'both'
    outcome_types: list[str] = field(default_factory=list)
    predictor_types: list[str] = field(default_factory=list)
    min_sample_size: int = 10
    assumptions: list[str] = field(default_factory=list)
    reference_impl: str = ""
    r_reference: str | None = None
    phase: int = 15
    extra: str | None = None
    default_params: dict[str, Any] = field(default_factory=dict)


# Registre interne (dictionnaire protégé)
_METHOD_REGISTRY: dict[str, tuple[MethodSpec, Callable[..., Any]]] = {}


class MethodRegistrationError(StatisticalIDEError):
    """Erreur levée lors d'une tentative d'enregistrement invalide."""
    pass


def register_method(spec: MethodSpec, func: Callable[..., Any]) -> None:
    """Enregistre une méthode statistique dans le registre central.

    Args:
        spec: La fiche d'identité de la méthode (MethodSpec).
        func: La fonction Python déterministe qui exécute la méthode.

    Raises:
        MethodRegistrationError: Si le method_id existe déjà dans le registre.
    """
    if spec.method_id in _METHOD_REGISTRY:
        raise MethodRegistrationError(
            f"La méthode '{spec.method_id}' est déjà enregistrée. "
            "Un identifiant de méthode doit être unique."
        )
    _METHOD_REGISTRY[spec.method_id] = (spec, func)


def get_method(method_id: str) -> tuple[MethodSpec, Callable[..., Any]]:
    """Récupère une méthode et sa fonction d'exécution par son identifiant.

    Args:
        method_id: L'identifiant unique de la méthode.

    Returns:
        Un tuple contenant (MethodSpec, fonction d'exécution).

    Raises:
        KeyError: Si la méthode n'est pas trouvée dans le registre.
    """
    if method_id not in _METHOD_REGISTRY:
        raise KeyError(
            f"Méthode '{method_id}' introuvable dans le registre. "
            "Vérifiez qu'elle a bien été enregistrée et que son extra est installé."
        )
    return _METHOD_REGISTRY[method_id]


def list_methods(family: str | None = None) -> list[MethodSpec]:
    """Liste les spécifications des méthodes disponibles.

    Ne déclenche aucun import de dépendance optionnelle (T18).

    Args:
        family: Si fourni, filtre les méthodes par famille statistique.

    Returns:
        Liste des MethodSpec correspondantes.
    """
    specs = [spec for spec, _ in _METHOD_REGISTRY.values()]
    if family is not None:
        specs = [spec for spec in specs if spec.family == family]
    return specs


def _register_v1_methods() -> None:
    """Enregistre les méthodes historiques de la V1 dans le registre.
    
    Cette fonction est appelée au démarrage de l'application pour assurer
    la rétrocompatibilité avec les méthodes déjà implémentées.
    """
    # Import local pour éviter les dépendances circulaires et respecter T18
    try:
        # Note: Les imports réels dépendront de l'arborescence exacte de la V1.
        # Ici nous utilisons des stubs (fonctions vides) pour que le registre
        # puisse être initialisé même si les modules V1 ne sont pas encore
        # refactorisés.
        from app.statistics import provider as v1_provider
        
        v1_methods_config = [
            ("chi_square", "Test du Khi-deux d'indépendance", "inférence", ["catégorielle"], ["catégorielle"], "scipy.stats.chi2_contingency", "stats::chisq.test"),
            ("t_test", "Test t de Student", "inférence", ["quantitative"], ["binaire"], "scipy.stats.ttest_ind", "stats::t.test"),
            ("anova", "Analyse de variance (ANOVA)", "inférence", ["quantitative"], ["catégorielle"], "scipy.stats.f_oneway", "stats::aov"),
            ("linear_regression", "Régression linéaire (OLS)", "régression", ["quantitative"], ["quantitative", "binaire", "catégorielle"], "statsmodels.api.OLS", "stats::lm"),
            ("logistic_regression", "Régression logistique binaire", "régression", ["binaire"], ["quantitative", "binaire", "catégorielle"], "statsmodels.api.Logit", "stats::glm"),
            ("multinomial_logistic_regression", "Régression logistique multinomiale", "régression", ["catégorielle"], ["quantitative", "binaire", "catégorielle"], "statsmodels.api.MNLogit", "nnet::multinom"),
            ("firth_logistic", "Régression logistique de Firth", "régression", ["binaire"], ["quantitative", "binaire", "catégorielle"], "firth", "logistf::logistf"),
            ("compare_models", "Comparaison de modèles (AIC/BIC/LRT)", "diagnostic", [], [], "statsmodels", "stats::anova"),
        ]
        
        for method_id, label, family, out_types, pred_types, ref_impl, r_ref in v1_methods_config:
            # Récupération de la fonction réelle si elle existe dans le provider V1
            func = getattr(v1_provider, f"run_{method_id}", lambda *args, **kwargs: None)
            
            spec = MethodSpec(
                method_id=method_id,
                label_fr=label,
                family=family,
                engine="both" if method_id == "firth_logistic" else "python",
                outcome_types=out_types,
                predictor_types=pred_types,
                min_sample_size=20 if method_id != "chi_square" else 5,
                assumptions=[],  # À compléter dans les phases spécifiques
                reference_impl=ref_impl,
                r_reference=r_ref,
                phase=1,  # Méthodes V1
                extra=None,
                default_params={}
            )
            
            # On ignore silencieusement si déjà enregistré (cas des redémarrages ou tests)
            if method_id not in _METHOD_REGISTRY:
                register_method(spec, func)
                
    except ImportError:
        # Si le provider V1 n'est pas encore disponible, on enregistre des stubs
        pass


# Initialisation automatique au premier import du module
_register_v1_methods()
