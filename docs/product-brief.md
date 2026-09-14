# Product Brief — Statistical IDE ("Cursor pour Statisticiens")  
   
## 1. Vision  
Créer un environnement de travail statistique agentique, fiable et transparent, qui agit comme un "Cursor pour les statisticiens". L'utilisateur importe ses données et son protocole de recherche (question, objectifs, hypothèses). Le système comprend le contexte, propose le plan d'analyse optimal, génère et exécute le code (Python/R), détecte et corrige les problèmes statistiques (séparation, multicolinéarité, manquants), et produit un rapport reproductible. L'objectif n'est pas de remplacer l'expert, mais de lui offrir un copilote qui gère la complexité technique tout en laissant le contrôle total sur le raisonnement, le code et les résultats.  
   
## 2. Cas d'usage réels  
   
### Cas 1 : Le chercheur avec un protocole défini  
- **Contexte** : Un utilisateur importe un dataset (ex: EDS) et un court protocole : "Je veux identifier les déterminants de l'anémie chez les enfants de moins de 5 ans, en contrôlant pour le niveau d'éducation de la mère et le milieu de résidence."  
- **Action du système** : Le système identifie la variable cible (anémie), les prédicteurs, propose une régression logistique, vérifie les hypothèses, détecte une éventuelle séparation complète, bascule automatiquement vers une régression de Firth si nécessaire, et génère le tableau de résultats + le code reproductible.  
- **Valeur ajoutée** : Gain de temps massif sur le choix du modèle et la gestion des cas limites, avec une traçabilité totale (comme dans un IDE).  
   
### Cas 2 : Le consultant statisticien en mode "Audit & Correction"  
- **Contexte** : Un consultant récupère une analyse faite par un étudiant sur un fichier Excel "sale". Le modèle ne converge pas ou les p-values sont aberrantes.  
- **Action du système** : Le consultant importe les données et demande : "Diagnostique pourquoi ce modèle échoue". Le système profile les données, identifie les valeurs manquantes non déclarées, les catégories rares, la multicolinéarité (VIF > 10), et propose un plan de correction étape par étape que le consultant peut valider ou modifier.  
- **Valeur ajoutée** : Transformation d'un processus de débogage manuel et fastidieux en un flux de travail structuré et vérifiable.  
   
### Cas 3 : L'exploration guidée ("Blind Spot Discovery")  
- **Contexte** : Un utilisateur a des données mais n'est pas sûr de la meilleure approche. Il demande : "Analyse les déterminants de cette variable et propose le modèle statistique approprié."  
- **Action du système** : Le système analyse la distribution de la variable cible (continue, binaire, catégorielle ?), propose 2-3 approches méthodologiques avec leurs hypothèses, exécute la plus robuste, affiche les diagnostics, et permet à l'utilisateur de "zoomer" sur le code généré pour le modifier manuellement si besoin.  
- **Valeur ajoutée** : Démocratisation de la rigueur statistique sans sacrifier la transparence méthodologique.  
   
## 3. Méthodes statistiques ciblées (Fondations du MVP)  
Le système doit être capable de proposer, exécuter et diagnostiquer ces 10 méthodes de base :  
1. Statistiques descriptives univariées et bivariées (avec gestion intelligente des manquants).  
2. Tableaux croisés et test du χ² d'indépendance.  
3. Test t de Student et ANOVA (comparaison de moyennes avec vérification des variances).  
4. Régression linéaire multiple (OLS) avec diagnostics (VIF, résidus).  
5. Régression logistique binaire (avec détection automatique de la séparation).  
6. Régression de Firth (comme alternative robuste en cas de séparation complète/quasi-séparation).  
7. Régression logistique multinomiale.  
8. Analyse factorielle (ACP pour variables continues, AFC pour catégorielles).  
9. Comparaison de modèles (AIC / BIC / Rapport de vraisemblance).  
10. Gestion des plans de sondage complexes (pondérations, strates) pour les données d'enquêtes type EDS/MICS.  
   
## 4. Datasets de démarrage (Pour tester la robustesse du "Cursor")  
1. **Base EDS (Enquête Démographique et de Santé)** : Dataset réel, volumineux, avec pondérations, étiquettes de variables complexes et catégories rares (idéal pour tester la détection de séparation et la régression de Firth).  
2. **Base MICS (Multiple Indicator Cluster Survey)** : Pour tester la gestion des structures de données d'enquêtes internationales, des valeurs manquantes spécifiques et des variables dérivées.  
3. **Fichier Excel "sale" de mémoire d'étudiant** : Données réelles avec fautes de frappe, codes de manquants incohérents (ex: "99", "NSP", " "), et colonnes mal typées. Test ultime du module de *data cleaning* et de *profiling* automatique.  
4. **Dataset de référence R (ex: `mtcars` ou `Titanic`)** : Pour les tests unitaires de validation numérique (comparaison systématique des sorties Python/R à une tolérance de 1e-6).  
   
## 5. Hors périmètre explicite (v1)  
- Pas de "boîte noire" : le système ne doit jamais exécuter de code ou prendre une décision méthodologique sans que l'utilisateur puisse voir le raisonnement, le code généré et les diagnostics sous-jacents.  
- Pas de modèles de Machine Learning prédictifs complexes (Random Forest, Deep Learning) : l'accent est mis sur l'**inférence statistique** et l'explicabilité, pas sur la prédiction pure.  
- Pas de collaboration multi-utilisateurs en temps réel (focus "local-first" et reproductibilité individuelle dans un premier temps).  
- Le LLM ne calcule **jamais** : il orchestre, propose et explique, mais le calcul est toujours délégué à des fonctions Python/R déterministes et testées.  
