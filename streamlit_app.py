"""
Point d'entrée pour Streamlit Cloud.

Ce fichier permet de lancer l'application depuis la racine du projet,
ce qui garantit que le dossier 'app/' est bien dans le PYTHONPATH.
"""
import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
root = Path(__file__).parent.resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Exécuter l'application Streamlit
import runpy
runpy.run_path("app/ui/streamlit_app.py", run_name="__main__")
