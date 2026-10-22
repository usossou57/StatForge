"""Pont Python-R pour l'exécution de scripts R.

Ce module permet d'appeler des scripts R depuis Python en passant des données
via JSON, et de récupérer les résultats. Utilisé pour les méthodes statistiques
non disponibles en Python (ex: régression de Firth).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

# Chemin vers les scripts R
_SCRIPTS_DIR = Path(__file__).parent / "scripts"


def call_r_script(script_name: str, payload: dict, timeout: int = 60) -> dict:
    """Exécute un script R avec les données fournies en JSON.

    Écrit le payload dans un fichier JSON temporaire, appelle le script R
    via subprocess, et retourne le résultat lu depuis le fichier de sortie JSON.

    Args:
        script_name: Nom du script R (ex: 'firth_logistic.R').
        payload: Dictionnaire de données à passer au script R.
        timeout: Temps maximum d'exécution en secondes (défaut 60).

    Returns:
        Dictionnaire contenant les résultats du script R.

    Raises:
        FileNotFoundError: Si le script R n'existe pas.
        RuntimeError: Si R n'est pas installé ou si le script échoue.
        subprocess.TimeoutExpired: Si le script dépasse le timeout.
    """
    script_path = _SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script R introuvable : {script_path}")

    # Vérifie que R est installé
    try:
        subprocess.run(
            ["Rscript", "--version"],
            capture_output=True,
            check=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            "R n'est pas installé ou n'est pas dans le PATH. "
            "Installe R avec : sudo apt install -y r-base r-base-dev"
        ) from e

    # Crée des fichiers temporaires pour l'entrée et la sortie
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = Path(tmpdir) / "input.json"
        output_file = Path(tmpdir) / "output.json"

        # Écrit le payload en JSON
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # Appelle le script R
        try:
            result = subprocess.run(
                ["Rscript", str(script_path), str(input_file), str(output_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Le script R a échoué avec le code {e.returncode}.\n"
                f"stdout: {e.stdout}\n"
                f"stderr: {e.stderr}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"Le script R a dépassé le timeout de {timeout} secondes."
            ) from e

        # Lit le résultat JSON
        if not output_file.exists():
            raise RuntimeError(
                f"Le script R n'a pas produit de fichier de sortie : {output_file}"
            )

        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)
