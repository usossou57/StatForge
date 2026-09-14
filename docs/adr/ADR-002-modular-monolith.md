# ADR-002: Architecture modulaire monolithique  
   
**Date** : 14 septembre 2026  
   
**Statut** : Accepté  
   
## Contexte  
   
Le projet Statistical IDE doit être développé par un développeur solo, déployé localement sur une machine CPU-only (~12 Go RAM), et évoluer progressivement sur 14 phases. L'architecture doit permettre :  
- Une maintenance simple par une seule personne  
- Un déploiement local sans infrastructure complexe  
- Une évolution progressive sans refonte majeure  
- Des tests unitaires isolés par module  
   
## Problème  
   
Comment structurer l'application pour qu'elle soit maintenable, testable et évolutive, sans introduire de complexité inutile pour un développeur solo ?  
   
## Options envisagées  
   
### Option 1 : Microservices  
**Description** : Découper l'application en services indépendants (Data Engine, Statistical Engine, AI Layer, UI) communiquant via API REST ou message queue.  
   
**Avantages** :  
- Séparation stricte des responsabilités  
- Déploiement indépendant de chaque service  
- Scalabilité horizontale possible  
   
**Inconvénients** :  
- Surcharge réseau entre services (latence, sérialisation)  
- Complexité de déploiement (Docker, orchestration)  
- Gestion de l'état partagé complexe (transactions distribuées)  
- Overkill pour un développeur solo et un usage local-first  
- Impossible à tester facilement sans infrastructure  
   
### Option 2 : Monolithe pur  
**Description** : Toute l'application dans un seul module Python, sans séparation architecturale.  
   
**Avantages** :  
- Simplicité extrême (un seul dossier, un seul déploiement)  
- Pas de surcharge réseau  
- Tests unitaires simples  
   
**Inconvénients** :  
- Couplage fort entre modules (difficile de tester isolément)  
- Évolution difficile (modifier une partie impacte tout)  
- Impossible de remplacer un module (ex: changer le moteur statistique)  
- Code difficile à maintenir à long terme (>10k lignes)  
   
### Option 3 : Monolithe modulaire (décision retenue)  
**Description** : Un seul déploiable (application Python unique), mais avec une séparation stricte des modules par responsabilité (app/data, app/statistics, app/diagnostics, app/agent, etc.). Chaque module a ses propres contrats de données (dataclasses/Pydantic) et ne communique avec les autres que via ces contrats.  
   
**Avantages** :  
- Simplicité de déploiement (un seul `uv run`)  
- Pas de surcharge réseau (tout en mémoire)  
- Modules testables isolément (app/statistics ne dépend pas de app/agent)  
- Évolution progressive (ajouter un module sans toucher aux autres)  
- Remplacement possible d'un module (ex: pont R en Phase 6)  
- Compatible avec les contraintes T1 (développeur solo), T3 (CPU-only), T4 (local-first), T10 (architecture simple)  
   
**Inconvénients** :  
- Nécessite une discipline stricte pour respecter les frontières entre modules  
- Impossible de déployer un module indépendamment (mais ce n'est pas un besoin)  
   
## Décision  
   
**Monolithe modulaire** est retenu.  
   
**Justification** :  
1. **Contrainte T1 (développeur solo)** : Un développeur seul ne peut pas maintenir une architecture microservices. Le monolithe modulaire offre la simplicité du monolithe avec la maintenabilité des modules isolés.  
2. **Contrainte T3/T4 (CPU-only, local-first)** : Pas de surcharge réseau, pas d'infrastructure externe. Tout tourne en local, en mémoire.  
3. **Contrainte T10 (architecture simple)** : Un seul déploiable, pas de queue, pas de base de données externe avant la Phase 11.  
4. **Évolution progressive** : Les 14 phases du projet ajoutent progressivement des modules (Phase 2 : data, Phase 3 : statistics, Phase 7 : agent, etc.). Le monolithe modulaire permet cette évolution sans refonte.  
5. **Testabilité** : Chaque module a ses propres contrats (Variable, AnalysisResult, DiagnosticFinding). Les tests unitaires peuvent valider un module isolément, sans dépendance aux autres.  
   
## Conséquences  
   
**Positives** :  
- Déploiement trivial (`uv run streamlit run app/ui/streamlit_app.py`)  
- Tests unitaires isolés par module  
- Évolution progressive sans refonte  
- Remplacement possible d'un module (ex: pont R)  
- Compatible avec toutes les contraintes techniques (T1-T15)  
   
**Négatives** :  
- Nécessite une discipline stricte pour respecter les frontières entre modules (pas d'import de app.agent dans app.statistics)  
- Impossible de déployer un module indépendamment (mais ce n'est pas un besoin pour un usage local-first)  
   
## Condition de révision  
   
Cette décision devrait être reconsidérée si :  
- Le projet devient multi-utilisateurs avec besoin de scalabilité horizontale (Phase 11+ ?)  
- Un module devient trop lourd et nécessite un déploiement indépendant (ex: moteur statistique sur un serveur dédié)  
- L'équipe passe de 1 développeur à 5+ (microservices deviendrait pertinent)  
