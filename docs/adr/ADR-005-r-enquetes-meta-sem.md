# ADR-005 : R pour enquêtes, méta-analyse, SEM et frailty

## Statut
Accepted

## Date
2026-10-05

## Contexte
La V2 vise une couverture statistique solide pour chercheurs et statisticiens.
Plusieurs familles de méthodes disposent d'implémentations R de référence,
largement validées par la communauté scientifique, mais d'équivalents Python
soit inexistants, soit immatures :

- **Enquêtes à plan complexe** : le package R `survey` est le standard
  mondial (utilisé par les instituts statistiques nationaux). L'équivalent
  Python n'existe pas à maturité équivalente.
- **Méta-analyse** : `metafor` (R) est la référence avec des estimateurs
  (REML, DL, etc.) et des diagnostics (funnel, Egger, trim-and-fill) très
  validés. `statsmodels` a un module de méta-analyse mais moins complet.
- **Modèles à équations structurelles (SEM/CFA)** : `lavaan` (R) est le
  standard. Pas d'équivalent Python mature.
- **Frailty dans les modèles de survie** : `coxme` et le terme `frailty()`
  de `survival` (R) sont les références. `lifelines` (Python) ne couvre
  pas ces cas.

Tenter de réimplémenter ces méthodes en Python pur :
- prendrait un temps de développement disproportionné pour un développeur solo ;
- produirait des résultats non validés par la communauté scientifique ;
- irait à l'encontre de la contrainte T7 (« ne pas réinventer les maths »)
  et T12 (validation contre R).

## Décision
Utiliser **R via le pont existant** (`app/r_bridge`, V1 Phase 6) pour ces
quatre familles. Les règles suivantes s'appliquent :

1. **Aucun `rpy2`** (T20) : tout appel à R passe par `call_r_script` ou
   `call_r_script_with_data` (nouvelle fonction pour les gros volumes,
   Phase 23), via des scripts `.R` dédiés et des échanges JSON/CSV.
2. **Dégradation propre** : si R n'est pas installé, les méthodes
   concernées sont masquées du catalogue (pas de crash, pas de message
   d'erreur brutal).
3. **Validation croisée** : chaque méthode via R est validée numériquement
   contre la sortie R de référence (tolérances documentées, V2 §6.3).
4. **Transparence** : le `MethodSpec` indique `engine="r"` et
   `r_reference` (ex. `"survey::svyglm"`). L'utilisateur voit quel
   moteur a été utilisé.
5. **Gros volumes** : pour les enquêtes, une fonction
   `call_r_script_with_data` écrit le DataFrame en CSV temporaire
   (pas en JSON, trop lourd) et passe les chemins à R.

Packages R à installer (uniquement quand la phase est atteinte) :
- Phase 23 : `survey`, `metafor`
- Phase 20 (SEM) : `lavaan`, `psych`, `FactoMineR`
- Phase 21 (frailty) : `survival`, `coxme`

## Options envisagées

### Option A : Réimplémenter en Python pur (rejetée)
- ✅ Pas de dépendance à R.
- ❌ Travail colossal, résultats non validés, non crédible scientifiquement.
- ❌ Contredit T7 et T12.

### Option B : R via le pont existant (retenue)
- ✅ Réutilise l'infrastructure V1 (pont R).
- ✅ Résultats validés par la communauté scientifique.
- ✅ Cohérent avec T7 (ne pas réinventer) et T12 (validation R).
- ❌ Dépendance à R (déjà acceptée en V1 Phase 6 pour Firth).

### Option C : Abandonner ces familles (rejetée)
- ✅ Pas de dépendance R supplémentaire.
- ❌ Couverture statistique incomplète, inacceptable pour des chercheurs.
- ❌ Contredit l'objectif de la V2 (§1.2).

## Conséquences
- **Positives** : couverture statistique complète, résultats validés,
  réutilisation du pont V1.
- **Négatives** : l'utilisateur doit installer R et les packages
  correspondants pour ces familles (documenté dans le README).
- **Tests** : tests marqués `@pytest.mark.r_required`, ignorés en CI
  sans R ; job CI-R hebdomadaire (V2 §4.5) pour les valider.

## Condition de révision
- Si un équivalent Python mature et validé émerge pour l'une de ces
  familles (ex. un package Python de SEM validé par la communauté).
- Si la maintenance du pont R devient trop coûteuse.
