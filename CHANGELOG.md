
---

## Fichier 2 : `CHANGELOG.md`

```markdown
# Changelog

Toutes les modifications notables de ce projet sont documentées dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

## [1.0.0] — 2026-10-05

### Ajouté

#### Phase 1 — Architecture & Conventions
- Règle architecturale : *IA PROPOSE → MOTEUR CALCULE → VALIDATEUR CONTRÔLE → UTILISATEUR DÉCIDE*
- Conventions de code (PEP 8, Ruff, docstrings Google en français)
- Contrats de données (`AnalysisResult`, `DiagnosticFinding`)

#### Phase 2 — Chargement des données
- Support des formats CSV, Excel (XLSX/XLS), SPSS (SAV via `pyreadstat`)
- Génération automatique du dictionnaire de données (types, modalités, hash SHA256)
- Profilage automatique (statistiques descriptives, valeurs manquantes)

#### Phase 3 — Modélisation des résultats
- `AnalysisResult` : objet structuré unique pour tous les résultats
- `DiagnosticFinding` : diagnostic avec sévérité (certain_issue, probable_issue, warning, information)

#### Phase 4 — Diagnostics statistiques
- Calculs déterministes pour : normalité, homoscédasticité, multicollinéarité, outliers, taille d'échantillon
- Intégration dans le flux d'analyse

#### Phase 5 — Registre d'outils
- `execute_tool()` : point d'entrée unique pour toutes les analyses
- Registre centralisé des méthodes statistiques

#### Phase 6 — Moteur statistique
- `run_chi_square` : test du χ² d'indépendance
- `run_t_test` : test t de Student
- `run_anova` : ANOVA à un facteur
- `run_linear_regression` : régression OLS (formules Patsy)
- `run_logistic_regression` : régression logistique binaire
- `run_multinomial_logistic_regression` : régression multinomiale

#### Phase 7 — Agent LLM
- `GroqProvider` : interface avec l'API Groq
- `build_analysis_plan()` : planification d'analyse via LLM
- `needs_clarification()` : détection des plans à faible confiance

#### Phase 8 — Orchestrator
- `propose_correction()` : proposition de corrections basée sur les diagnostics
- `execute_confirmed_action()` : exécution après confirmation humaine

#### Phase 9 — Contraintes T8/T13/T14
- T8 : aucun DataFrame brut n'est envoyé au LLM
- T13 : diagnostics toujours calculés par du code déterministe
- T14 : confirmation humaine obligatoire pour les corrections

#### Phase 10 — Interface utilisateur
- Page Upload & Dictionnaire
- Page Analyse (question, plan, exécution, diagnostics, corrections)
- Page Rapport (HTML/Markdown, graphiques, téléchargement)
- Navigation robuste avec `session_state`

#### Phase 11 — Reproductibilité
- `save_analysis()` / `load_project()` : persistance JSON
- `replay_analysis()` : rejeu via `execute_tool`
- Historique persistant (`projects/history.jsonl`)
- Compteur de rejeux par projet
- **11 tests automatisés** couvrant 6 méthodes statistiques

#### Phase 12 — Tests utilisateurs
- Phase documentée mais non réalisée (manque de testeurs)
- Formulaire de retour structuré prévu

#### Phase 13 — Sécurité
- `validate_generated_code()` : analyse AST, liste blanche d'imports
- `run_sandboxed()` : exécution isolée avec limites CPU/mémoire
- **20 tests** de sécurité (validation, timeout, boucles infinies, sur-allocation)

#### Phase 14 — Intégration sandbox dans l'UI
- Bouton "Exécuter le code reproductible" dans la page Rapport
- Préfixage automatique du code avec le chargement du fichier source
- Affichage des résultats dans l'interface
- Blocage des codes malveillants avec message explicite

### Corrigé
- Gestion des noms de variables avec espaces (syntaxe `Q('nom')` de Patsy)
- Synchronisation du compteur de rejeux entre les sections de l'UI
- Remplacement intelligent du placeholder `CHEMIN_VERS_VOS_DONNEES.*` selon l'extension du fichier
- Gestion robuste de `chi_square` dans `replay_analysis` (combinaison `outcome_variable` + `predictor_variables`)
- Navigation sidebar avec `key` dynamique pour éviter les mismatch Unicode

### Sécurité
- Sandbox isolé (subprocess)
- Validation AST stricte (imports, `eval`, `exec`, `open` en écriture, appels système)
- Limites de ressources : timeout 15s, mémoire 512 Mo

## [0.1.0] — 2026-10-03

### Ajouté
- Structure initiale du projet
- Premières fonctions statistiques
- Interface Streamlit basique

---

## Notes de version

### v1.0.0
Première version publique de StatForge. MVP fonctionnel couvrant l'ensemble du workflow : upload → analyse → rapport → reproductibilité → sécurité.

**Statistiques** :
- ~3 500 lignes de code Python
- 31 tests automatisés
- 6 méthodes statistiques
- 3 formats de fichiers supportés
- 1 fournisseur LLM (Groq)
