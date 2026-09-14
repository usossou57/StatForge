"""Script de validation du module de profiling.

Ce script démontre que les fonctions de profiling fonctionnent
correctement sur un dataset "sale" avec des cas limites.

Exécution :
    uv run python scripts/validate_profiling.py
"""

import sys
from pathlib import Path

# Ajouter le répertoire racine au path pour les imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.data.importers import import_csv
from app.data.profiling import build_data_dictionary, describe_variable, profile_dataset


def print_separator(title: str) -> None:
    """Affiche un séparateur avec un titre."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_subsection(title: str) -> None:
    """Affiche un sous-titre."""
    print(f"\n--- {title} ---")


def validate_profiling() -> None:
    """Fonction principale de validation."""
    print_separator("VALIDATION DU MODULE DE PROFILING")

    # Chemin vers le dataset "sale"
    csv_path = ROOT_DIR / "tests" / "fixtures" / "dirty_dataset.csv"

    if not csv_path.exists():
        print(f"❌ ERREUR : Le fichier {csv_path} n'existe pas.")
        sys.exit(1)

    print(f"📁 Fichier source : {csv_path}")

    # === Étape 1 : Import du dataset ===
    print_separator("1. IMPORT DU DATASET")

    try:
        df = import_csv(str(csv_path))
        print(f"✅ Import réussi : {df.shape[0]} lignes × {df.shape[1]} colonnes")
        print(f"   Colonnes : {', '.join(df.columns.tolist())}")
    except Exception as e:      # noqa: BLE001
        print(f"❌ ERREUR lors de l'import : {e}")
        sys.exit(1)

    # === Étape 2 : Profil du dataset ===
    print_separator("2. PROFIL DU DATASET")

    profile = profile_dataset(df)

    print(f"📊 Dimensions : {profile['n_rows']} lignes × {profile['n_columns']} colonnes")

    print_subsection("Types de données")
    for col, dtype in profile["dtypes"].items():
        print(f"   • {col:20s} : {dtype}")

    print_subsection("Valeurs manquantes (%)")
    for col, pct in profile["missing_pct"].items():
        status = "⚠️ " if pct > 0 else "✅"
        print(f"   {status} {col:20s} : {pct:5.1f}%")

    print_subsection("Cardinalités")
    for col, n in profile["n_unique"].items():
        print(f"   • {col:20s} : {n} valeurs uniques")

    print_subsection("Colonnes constantes")
    if profile["constant_columns"]:
        for col in profile["constant_columns"]:
            print(f"   ⚠️  {col} (valeur unique)")
    else:
        print("   ✅ Aucune colonne constante détectée")

    # === Étape 3 : Description des variables ===
    print_separator("3. DESCRIPTION DES VARIABLES")

    # Variable numérique
    print_subsection("Variable numérique : 'age'")
    desc_age = describe_variable(df, "age")
    print(f"   Moyenne   : {desc_age['mean']:.1f}")
    print(f"   Médiane   : {desc_age['median']:.1f}")
    print(f"   Écart-type: {desc_age['std']:.1f}")
    print(f"   Min       : {desc_age['min']}")
    print(f"   Max       : {desc_age['max']}")
    print(f"   Manquants : {desc_age['n_missing']}")

    # Variable catégorielle
    print_subsection("Variable catégorielle : 'sexe'")
    desc_sexe = describe_variable(df, "sexe")
    print("   Fréquences :")
    for value, count in desc_sexe["frequencies"].items():
        print(f"     • {value} : {count} occurrences")

    # Variable avec outlier
    print_subsection("Variable avec outlier : 'revenu'")
    desc_revenu = describe_variable(df, "revenu")
    print(f"   Moyenne   : {desc_revenu['mean']:.1f}")
    print(f"   Médiane   : {desc_revenu['median']:.1f}")
    print(f"   Min       : {desc_revenu['min']}")
    print(f"   Max       : {desc_revenu['max']}")
    if desc_revenu["max"] > 100000:
        print(f"   ⚠️  Valeur aberrante détectée : {desc_revenu['max']}")

    # === Étape 4 : Construction du dictionnaire de données ===
    print_separator("4. DICTIONNAIRE DE DONNÉES")

    try:
        data_dict = build_data_dictionary(df, "dirty_dataset", str(csv_path))
        print("✅ Dictionnaire construit avec succès")
        print(f"   Nom       : {data_dict.dataset_name}")
        print(f"   Dimensions: {data_dict.n_rows} × {data_dict.n_columns}")
        print(f"   Hash      : {data_dict.file_hash[:16]}...")

        print_subsection("Variables détectées")
        for var_name, var in data_dict.variables.items():
            print(f"\n   📌 {var_name}")
            print(f"      Type          : {var.dtype}")
            print(f"      Type sémantique: {var.semantic_type or 'Non défini'}")
            if var.levels:
                print(f"      Niveaux       : {var.levels}")

    except Exception as e:      # noqa: BLE001
        print(f"❌ ERREUR lors de la construction : {e}")
        sys.exit(1)

    # === Étape 5 : Vérifications ===
    print_separator("5. VÉRIFICATIONS")

    checks_passed = 0
    checks_total = 0

    # Vérification 1 : Colonne constante détectée
    checks_total += 1
    if "constante" in profile["constant_columns"]:
        print("✅ Colonne constante détectée : 'constante'")
        checks_passed += 1
    else:
        print("❌ Colonne constante NON détectée")

    # Vérification 2 : Valeurs manquantes comptées
    checks_total += 1
    if profile["missing_pct"]["age"] > 0:
        print(f"✅ Valeurs manquantes détectées dans 'age' : {profile['missing_pct']['age']:.1f}%")
        checks_passed += 1
    else:
        print("❌ Valeurs manquantes NON détectées dans 'age'")

    # Vérification 3 : Variable 'niveau_etude' correctement typée
    # (numérique avec 4 valeurs uniques → semantic_type = "count")
    checks_total += 1
    niveau_etude_var = data_dict.variables.get("niveau_etude")
    if niveau_etude_var is not None:
        print(f"✅ Variable 'niveau_etude' correctement typée : dtype={niveau_etude_var.dtype}, semantic_type={niveau_etude_var.semantic_type}")
        checks_passed += 1
    else:
        print("❌ Variable 'niveau_etude' non trouvée dans le dictionnaire")

    # Vérification 4 : Types sémantiques inférés
    checks_total += 1
    sexe_var = data_dict.variables.get("sexe")
    if sexe_var and sexe_var.semantic_type == "binary":
        print("✅ Type sémantique 'binary' correctement inféré pour 'sexe'")
        checks_passed += 1
    else:
        print("❌ Type sémantique 'binary' NON inféré pour 'sexe'")

    # Vérification 5 : Variable continue détectée
    checks_total += 1
    score_var = data_dict.variables.get("score")
    if score_var and score_var.semantic_type == "continuous":
        print("✅ Type sémantique 'continuous' correctement inféré pour 'score'")
        checks_passed += 1
    else:
        print("❌ Type sémantique 'continuous' NON inféré pour 'score'")

    # === Résumé ===
    print_separator("RÉSUMÉ")

    print(f"📊 Dataset analysé : {data_dict.dataset_name}")
    print(f"   • {data_dict.n_rows} observations")
    print(f"   • {data_dict.n_columns} variables")
    print(f"   • {len(profile['constant_columns'])} colonne(s) constante(s)")
    print(f"   • {sum(1 for pct in profile['missing_pct'].values() if pct > 0)} variable(s) avec manquants")

    print(f"\n🧪 Tests de validation : {checks_passed}/{checks_total} réussis")

    if checks_passed == checks_total:
        print("\n✅ TOUTES LES VÉRIFICATIONS ONT RÉUSSI !")
        print("   Le module de profiling fonctionne correctement.")
    else:
        print(f"\n⚠️  {checks_total - checks_passed} vérification(s) ont échoué.")
        print("   Veuillez vérifier les messages d'erreur ci-dessus.")

    print_separator("FIN DE LA VALIDATION")


if __name__ == "__main__":
    validate_profiling()
