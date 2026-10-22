"""Validation de l'enregistrement des méthodes de la phase 18 dans le registre."""
from app.statistics.registry import list_methods


def main():
    # Méthodes attendues de la phase 18
    expected_methods = [
        # T18.1 - GLM
        "probit_regression",
        "poisson_regression",
        "negative_binomial_regression",
        "gamma_regression",
        # T18.2 - Ordinale
        "ordinal_regression",
        # T18.3 - Régularisée
        "ridge_regression",
        "lasso_regression",
        "elastic_net_regression",
        # T18.4 - Robuste/Polynomiale/Splines
        "robust_regression",
        "polynomial_regression",
        "spline_regression",
    ]
    
    print("=== Validation du registre pour la phase 18 ===\n")
    
    all_registered = list_methods()
    registered_ids = {m.method_id for m in all_registered}
    
    missing = []
    for method_id in expected_methods:
        if method_id in registered_ids:
            spec = next(m for m in all_registered if m.method_id == method_id)
            print(f"✅ {method_id:40s} - {spec.label_fr}")
        else:
            print(f"❌ {method_id:40s} - NON ENREGISTRÉ")
            missing.append(method_id)
    
    print(f"\n=== Résumé ===")
    print(f"Méthodes attendues: {len(expected_methods)}")
    print(f"Méthodes enregistrées: {len(expected_methods) - len(missing)}")
    print(f"Méthodes manquantes: {len(missing)}")
    
    if missing:
        print(f"\n⚠️  Méthodes manquantes: {missing}")
        return False
    else:
        print("\n🎉 Toutes les méthodes de la phase 18 sont correctement enregistrées !")
        return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)