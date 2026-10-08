# ADR-003 : Registre de méthodes plutôt qu'extension du Protocol

## Statut
Accepted

## Date
2026-10-05

## Contexte
En V1, le `StatisticalEngineProvider` (Protocol) listait explicitement chaque méthode
(`run_chi_square`, `run_t_test`, `run_linear_regression`, etc.). Avec l'extension V2
qui vise ~100 méthodes réparties sur 15 familles (inférence, régression, survie,
séries temporelles, causalité, ML, etc.), cette approche devient impraticable :
- Le Protocol deviendrait un fichier de plusieurs centaines de lignes.
- Chaque nouvelle méthode obligerait à modifier l'interface centrale, ce qui casse
  le principe de séparation des familles.
- L'UI et le planner auraient du mal à découvrir dynamiquement les méthodes
  disponibles.

## Décision
Mettre en place un **registre central de méthodes** (`app/statistics/registry.py`)
basé sur une dataclass `MethodSpec` qui décrit chaque méthode (identifiant, libellé,
famille, moteur, types de variables acceptés, hypothèses, référence Python/R, etc.).

Chaque module de méthode s'enregistre lui-même au moment de son import via
`register_method(spec, func)`. L'UI, le planner et l'agent interrogent le registre
via `list_methods()` et `get_method()`.

Le `StatisticalEngineProvider` de la V1 est conservé tel quel pour la
rétrocompatibilité, mais il n'est plus le point d'entrée unique pour les nouvelles
méthodes.

## Options envisagées

### Option A : Allonger le Protocol (rejetée)
Ajouter chaque méthode au `StatisticalEngineProvider`.
- ✅ Simple, cohérent avec la V1.
- ❌ Fichier central ingérable à ~100 méthodes.
- ❌ Chaque nouvelle famille impose une modification de l'interface.
- ❌ Découverte dynamique impossible sans importer tout le module.

### Option B : Registre central par `MethodSpec` (retenue)
- ✅ Chaque famille est autonome (son module s'auto-enregistre).
- ✅ Découverte dynamique pour l'UI et le planner.
- ✅ Métadonnées riches (hypothèses, référence R, engine) exploitables.
- ✅ Compatible avec les dépendances optionnelles (extras).
- ❌ Nécessite une discipline : toute méthode doit s'enregistrer.

### Option C : Plugin system complet (rejetée)
Marché de plugins externes.
- ✅ Très flexible.
- ❌ Hors périmètre V2 (§1.3 : pas de marché de plugins).
- ❌ Complexité inutile pour un développeur solo.

## Conséquences
- **Positives** : architecture modulaire, UI/planner agnostiques, ajout de méthode
  sans toucher au cœur.
- **Négatives** : discipline d'enregistrement à respecter (testée en CI via un
  compteur de méthodes).
- **Migration** : les méthodes V1 (χ², t, ANOVA, OLS, logistique, MNLogit, Firth,
  compare_models) sont enregistrées dans le registre sans modifier leurs signatures.

## Condition de révision
- Si le nombre de méthodes dépasse 200 et que des regroupements par sous-famille
  deviennent nécessaires.
- Si un besoin de plugins externes émerge (à traiter en V3, pas en V2).
