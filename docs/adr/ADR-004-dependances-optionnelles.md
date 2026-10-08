# ADR-004 : Dépendances optionnelles par extras

## Statut
Accepted

## Date
2026-10-05

## Contexte
La V2 ajoute de nombreuses familles de méthodes (survie, séries temporelles,
multivarié, ML, bayésien, etc.) qui nécessitent des bibliothèques Python lourdes :
- `lifelines` (survie)
- `linearmodels`, `arch` (séries temporelles, panel)
- `factor-analyzer`, `prince` (multivarié)
- `shap`, `imbalanced-learn` (ML)
- `pymc`, `arviz` (bayésien)
- `python-docx`, `weasyprint` (reporting)

Installer toutes ces dépendances par défaut :
- alourdirait l'installation de base (plusieurs centaines de Mo, voire Go avec
  `pymc` et ses compilations) ;
- ralentirait la CI et les tests de base ;
- imposerait à l'utilisateur des dépendances dont il n'a pas besoin.

À l'inverse, ne rien installer et laisser l'utilisateur gérer manuellement
créerait une mauvaise expérience (erreurs `ImportError` obscures au moment de
l'exécution).

## Décision
Utiliser le mécanisme d'**extras optionnels** de `uv`/`pip`
(`uv add --optional <extra> <paquet>`).

Les extras déclarés sont :
- `multivariate` : factor-analyzer, prince
- `embedding` : umap-learn (séparé car tire `numba`, très lourd)
- `survival` : lifelines
- `timeseries` : linearmodels, arch
- `causal` : rdrobust, networkx
- `ml` : shap, imbalanced-learn
- `reporting` : python-docx, nbformat, weasyprint
- `bayes` : pymc, arviz
- `design` : pyDOE3

Règles d'implémentation :
1. **Imports paresseux** (T18) : les bibliothèques optionnelles sont importées
   **à l'intérieur** des fonctions qui les utilisent, jamais au niveau du module.
   L'application et les tests de base doivent démarrer sans elles.
2. **Erreur explicite** (T19) : si une méthode est appelée sans que son extra
   soit installé, elle lève `DependencyMissingError` avec le message exact
   d'installation (ex. « uv add --optional survival lifelines »).
3. **CI complète** : la CI exécute `uv sync --all-extras --dev` pour tester
   l'ensemble, mais les tests de base passent sans extras.
4. **Filtrage UI/planner** : les méthodes dont l'extra est absent sont
   simplement masquées du catalogue (pas de message d'erreur).

## Options envisagées

### Option A : Tout installer par défaut (rejetée)
- ✅ Simple pour l'utilisateur.
- ❌ Installation de base très lourde.
- ❌ Contredit la contrainte T3 (CPU-only, machine modeste).

### Option B : Extras optionnels (retenue)
- ✅ Installation de base légère.
- ✅ Mécanisme standard Python (pip/uv).
- ✅ Messages d'erreur clairs et actionnables.
- ❌ Demande à l'utilisateur de comprendre les extras.

### Option C : Détection automatique + installation à la volée (rejetée)
- ✅ Expérience "magique".
- ❌ Modifie l'environnement sans consentement explicite.
- ❌ Problèmes de reproductibilité et de sécurité.

## Conséquences
- **Positives** : installation légère, choix explicite de l'utilisateur,
  messages d'erreur clairs.
- **Négatives** : documentation à produire pour expliquer les extras.
- **Tests** : la CI tourne avec tous les extras ; les tests locaux de base
  tournent sans.

## Condition de révision
- Si un extra devient si fréquent qu'il mérite d'être dans le noyau
  (ex. si >80 % des utilisateurs installent `survival`).
- Si un conflit de versions entre extras devient ingérable.
