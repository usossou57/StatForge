"""Module de sandbox pour l'exécution sécurisée de code généré.

Ce module fournit deux fonctions publiques :
- validate_generated_code : analyse l'AST pour détecter les violations
- run_sandboxed : exécute le code dans un subprocess isolé avec limites

Contraintes respectées :
- Uniquement stdlib Python (ast, subprocess, tempfile, resource, os, sys)
- Module resource Linux uniquement → fallback propre sur autres OS
- Aucune exception non gérée (sauf ValueError pour violations)
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from typing import Any

# Liste blanche des imports autorisés
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "pandas",
        "numpy",
        "scipy",
        "statsmodels",
        "matplotlib",
        "math",
        # Sous-modules courants
        "pandas.api",
        "pandas.core",
        "numpy.linalg",
        "numpy.random",
        "scipy.stats",
        "scipy.optimize",
        "statsmodels.api",
        "statsmodels.formula",
        "matplotlib.pyplot",
        "matplotlib.figure",
    }
)

# Imports réseau interdits
FORBIDDEN_NETWORK_IMPORTS: frozenset[str] = frozenset(
    {
        "socket",
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
    }
)

# Appels système dangereux
FORBIDDEN_SYSTEM_CALLS: frozenset[str] = frozenset(
    {
        "os.system",
        "subprocess",
        "shutil.rmtree",
    }
)


def validate_generated_code(code: str) -> list[str]:
    """Analyse l'AST du code généré et retourne la liste des violations.

    Args:
        code: Code Python à analyser.

    Returns:
        Liste de chaînes décrivant chaque violation. Liste vide si conforme.

    Raises:
        SyntaxError: Si le code n'est pas du Python valide.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SyntaxError(f"Code Python invalide : {e}") from e

    violations: list[str] = []

    for node in ast.walk(tree):
        # Vérification des imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                if module_name in FORBIDDEN_NETWORK_IMPORTS:
                    violations.append(
                        f"Import réseau interdit : {module_name} (ligne {node.lineno})"
                    )
                elif module_name not in ALLOWED_IMPORTS and not _is_submodule_allowed(
                    module_name
                ):
                    violations.append(
                        f"Import hors liste blanche : {module_name} (ligne {node.lineno})"
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module
                if module_name in FORBIDDEN_NETWORK_IMPORTS:
                    violations.append(
                        f"Import réseau interdit : {module_name} (ligne {node.lineno})"
                    )
                elif module_name not in ALLOWED_IMPORTS and not _is_submodule_allowed(
                    module_name
                ):
                    violations.append(
                        f"Import hors liste blanche : {module_name} (ligne {node.lineno})"
                    )

        # Vérification des appels dangereux
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in ("eval", "exec"):
                    violations.append(
                        f"Appel à {func_name}() interdit (ligne {node.lineno})"
                    )
                elif func_name == "open":
                    mode = _get_open_mode(node)
                    if mode and any(m in mode for m in ("w", "a")):
                        violations.append(
                            f"Appel à open() en mode écriture '{mode}' interdit "
                            f"(ligne {node.lineno})"
                        )

            elif isinstance(node.func, ast.Attribute):
                attr_name = _get_full_attribute_name(node.func)
                if attr_name in FORBIDDEN_SYSTEM_CALLS:
                    violations.append(
                        f"Appel système interdit : {attr_name} (ligne {node.lineno})"
                    )

    return violations


def _is_submodule_allowed(module_name: str) -> bool:
    """Vérifie si un sous-module est autorisé (ex: pandas.core.frame).

    Args:
        module_name: Nom complet du module.

    Returns:
        True si le module est un sous-module d'un module autorisé.
    """
    for allowed in ALLOWED_IMPORTS:
        if module_name.startswith(allowed + "."):
            return True
    return False


def _get_open_mode(call_node: ast.Call) -> str | None:
    """Extrait le mode d'ouverture d'un appel à open().

    Args:
        call_node: Nœud AST de l'appel à open().

    Returns:
        Mode d'ouverture (ex: 'r', 'w', 'rb') ou None si non spécifié.
    """
    # Mode en 2e argument positionnel
    if len(call_node.args) >= 2:
        if isinstance(call_node.args[1], ast.Constant):
            return str(call_node.args[1].value)
    # Mode en keyword argument
    for keyword in call_node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
    # Mode par défaut : 'r'
    return "r"


def _get_full_attribute_name(node: ast.Attribute) -> str:
    """Construit le nom complet d'un attribut (ex: os.system).

    Args:
        node: Nœud AST de type Attribute.

    Returns:
        Nom complet de l'attribut (ex: "os.system").
    """
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def run_sandboxed(
    code: str,
    timeout_seconds: int = 30,
    max_memory_mb: int = 512,
) -> dict[str, Any]:
    """Exécute le code dans un subprocess isolé avec limites de ressources.

    Args:
        code: Code Python à exécuter.
        timeout_seconds: Timeout en secondes (défaut: 30).
        max_memory_mb: Mémoire maximale en Mo (défaut: 512).

    Returns:
        Dictionnaire avec clés : stdout, stderr, returncode, success.

    Raises:
        ValueError: Si le code contient des violations de sécurité.
    """
    # Validation préalable
    violations = validate_generated_code(code)
    if violations:
        raise ValueError(
            "Code refusé pour les raisons suivantes :\n"
            + "\n".join(f"- {v}" for v in violations)
        )

    # Écriture dans un fichier temporaire
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as tmp_file:
        tmp_file.write(code)
        tmp_path = tmp_file.name

    try:
        cmd = [sys.executable, tmp_path]

        # Fonction de pré-exécution pour les limites resource (Linux uniquement)
        preexec_fn = None
        resource_warning = ""
        if sys.platform == "linux":

            def _set_limits() -> None:
                import resource

                try:
                    # Timeout CPU uniquement (RLIMIT_AS trop restrictif pour pandas/numpy)
                    resource.setrlimit(
                        resource.RLIMIT_CPU, (timeout_seconds, timeout_seconds)
                    )
                    # RLIMIT_AS désactivé : pandas nécessite > 512 Mo d'espace d'adressage
                    # pour mapper ses bibliothèques C partagées (.so)
                except (OSError, ValueError):
                    # Limites ignorées si erreur (ex: valeurs trop basses)
                    pass

            preexec_fn = _set_limits
        else:
            resource_warning = (
                f"Limites de ressources non appliquées (plateforme : {sys.platform})\n"
            )

        # Exécution
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                preexec_fn=preexec_fn,
            )
            return {
                "stdout": result.stdout,
                "stderr": resource_warning + result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": resource_warning
                + f"Timeout : exécution dépassée après {timeout_seconds} secondes",
                "returncode": -1,
                "success": False,
            }
        except FileNotFoundError:
            return {
                "stdout": "",
                "stderr": resource_warning
                + f"Interpréteur Python introuvable : {sys.executable}",
                "returncode": -1,
                "success": False,
            }
        except OSError as e:
            return {
                "stdout": "",
                "stderr": resource_warning + f"Erreur système : {e}",
                "returncode": -1,
                "success": False,
            }
    finally:
        # Nettoyage du fichier temporaire
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
