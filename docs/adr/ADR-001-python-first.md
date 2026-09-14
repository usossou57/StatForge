# ADR-001: Python-first avec pont R optionnel  
   
**Date** : 14 septembre 2026  
   
**Statut** : Accepté  
   
## Contexte  
   
Le projet Statistical IDE nécessite un moteur de calcul statistique robuste et fiable. Le choix du langage principal impacte directement :  
- La capacité à intégrer des modèles de langage (LLM) pour l'assistance IA  
- La richesse de l'écosystème de bibliothèques statistiques disponibles  
- La facilité de déploiement sur une machine CPU-only avec ~12 Go RAM  
- La maintenabilité par un développeur solo  
   
## Problème  
   
Quel langage choisir comme moteur principal de calcul statistique, sachant que R est historiquement la référence en statistiques académiques, mais que Python domine l'écosystème IA/ML ?  
   
## Options envisagées  
   
### Option 1 : R pur  
**Description** : Utiliser R comme langage unique, avec des bindings Python via rpy2.  
   
**Avantages** :  
- Référence historique en statistiques académiques  
- Packages statistiques extrêmement matures (logistf, survey, etc.)  
- Validation facile des résultats (comparaison directe)  
   
**Inconvénients** :  
- Intégration LLM moins naturelle (écosystème Python dominant pour l'IA)  
- Gestion des dépendances plus complexe (CRAN vs pip/uv)  
- Moins de bibliothèques pour le parsing de données modernes (Excel, JSON)  
- Courbe d'apprentissage plus raide pour un développeur solo  
   
### Option 2 : Python pur (sans R)  
**Description** : Implémenter toutes les méthodes statistiques en Python uniquement, sans pont R.  
   
**Avantages** :  
- Écosystème IA/ML natif (LangChain, OpenAI, Anthropic, Ollama)  
- Gestion de projet moderne avec uv, typage strict, outils de qualité  
- Parsing de données unifié (pandas, pyreadstat)  
- Déploiement simplifié (un seul langage)  
   
**Inconvénients** :  
- Certaines méthodes statistiques de pointe moins disponibles en Python (ex: régression de Firth)  
- Impossible de valider directement contre R (référence académique)  
- Réimplémenter des algorithmes existants = risque d'erreurs  
   
### Option 3 : Python-first avec pont R optionnel (décision retenue)  
**Description** : Python comme langage principal, avec un pont R (app/r_bridge) introduit en Phase 6 pour les méthodes spécifiques où R excelle.  
   
**Avantages** :  
- Bénéficie de l'écosystème Python pour l'IA et le parsing  
- Accès aux méthodes R de pointe via le pont (Firth, plans de sondage complexes)  
- Validation croisée possible (Python vs R)  
- Dégradation propre : si R n'est pas installé, le système fonctionne toujours avec les méthodes Python  
   
**Inconvénients** :  
- Complexité supplémentaire (gestion de deux langages)  
- Nécessite d'installer R en Phase 6 (optionnel)  
- Maintenance de scripts R en parallèle du code Python  
   
## Décision  
   
**Python-first avec pont R optionnel** est retenu.  
   
**Justification** :  
1. **Priorité à l'IA** : Le projet est un "Cursor pour statisticiens", l'intégration LLM est centrale. Python est le langage natif de l'IA.  
2. **Simplicité de déploiement** : Un développeur solo sur une machine CPU-only a besoin d'un setup minimal. Python + uv est plus simple que R + rpy2.  
3. **Pragmatisme** : 90% des méthodes statistiques courantes (régression, ANOVA, chi²) sont parfaitement implémentées dans statsmodels/scipy. Le pont R n'est nécessaire que pour 10% des cas (Firth, plans de sondage complexes).  
4. **Évolutivité** : Le pont R est introduit en Phase 6, après que le moteur Python soit validé. Cela permet de livrer un MVP fonctionnel sans R, puis d'ajouter la puissance de R si nécessaire.  
   
## Conséquences  
   
**Positives** :  
- MVP livrable rapidement sans dépendance à R  
- Écosystème Python riche pour l'IA et le parsing  
- Validation croisée possible (Python vs R) pour les méthodes critiques  
- Dégradation propre : le système fonctionne même sans R installé  
   
**Négatives** :  
- Complexité supplémentaire en Phase 6 (pont R)  
- Nécessité de maintenir des scripts R en parallèle  
- Tests d'intégration plus complexes (skipif R non installé)  
   
## Condition de révision  
   
Cette décision devrait être reconsidérée si :  
- L'écosystème Python rattrape R sur les méthodes statistiques de pointe (ex: régression de Firth devient stable dans firthlogist)  
- Les besoins en plans de sondage complexes deviennent prédominants (>50% des cas d'usage)  
- Un changement majeur dans l'écosystème LLM rend R plus attractif pour l'IA (peu probable)  
