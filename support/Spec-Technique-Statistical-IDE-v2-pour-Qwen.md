# Spécification technique — Statistical IDE, Extension V2

**Document de contexte pour génération de code assistée (Qwen Chat) — complément de la spécification V1**

## Sommaire

0. Comment utiliser ce document (0.1 pour toi, 0.2 pour Qwen Chat)
1. Contexte et périmètre de la V2
2. Règle architecturale (rappel) et règles ajoutées
3. Contraintes techniques (héritées T1–T15 + nouvelles T16–T24)
4. Stack technique et dépendances par phase
5. Organisation du projet (arborescence cible V2)
6. Conventions de code (compléments à la V1)
7. Contrats de données et interfaces nouveaux
8. Gabarit de prompt pour Qwen
9. Guide d'implémentation par phase (9.0 table de suivi + phases 15 à 29)
10. Annexe — commandes utiles
11. Annexe — checklist de validation globale V2

> **Numérotation** : les sections de ce document portent le même numéro que leur équivalent dans la V1 (0 à 11). Quand le texte écrit « V1 §7 », il s'agit de la section 7 de la spécification V1 ; « V2 §7 » désigne la section 7 de ce document.

---

## 0. Comment utiliser ce document

Ce document **prolonge** la spécification V1 (phases 0 à 14, toutes terminées). Il ne la remplace pas : les deux fichiers doivent rester ensemble dans les sources du projet Qwen Chat. La V1 reste la référence pour tout ce qui existe déjà ; la V2 ajoute les phases 15 à 29.

### 0.1 Pour toi (l'humain)

Le workflow est le même qu'en V1 :

1. Dépose ce fichier dans les sources du projet Qwen Chat, à côté de la V1.
2. Demande : *« où en est-on sur la V2 ? »* ou *« par quoi on commence ? »*.
3. Une fois la tâche identifiée, demande : *« donne-moi le code pour la tâche T17.2 »*.
4. Place le code au chemin indiqué, lance les tests (`uv run pytest`, `uv run ruff check .`, `uv run mypy app/`).
5. Dis à Qwen Chat que la tâche est faite et coche toi-même la case (⬜→✅) dans **ta** copie de la table 9.0.
6. Toute proposition de modifier un contrat de la V1 §7 ou de la V2 §7 est un **signal d'alerte**, jamais une suggestion à accepter automatiquement.

**Ce qui change par rapport à la V1 :**

- Il y a beaucoup plus de tâches (une centaine). Avance **phase par phase** et ne démarre pas une phase avant d'avoir validé la précédente.
- Chaque nouvelle méthode statistique suit la **Définition de Done commune** (V2 §9.1). Si une méthode est livrée sans test de validation contre R, la tâche n'est pas terminée.
- Plusieurs tâches sont **pour toi** (ADR, installation de R et de ses packages, jeux de données de référence). Elles sont marquées « (humain) » dans la table.
- Des **jalons** (V2-alpha, V2-beta, V2-RC) permettent de s'arrêter proprement : tu peux publier une version utile sans attendre la phase 29.

**Limite à vérifier :** comme en V1, je ne connais pas avec certitude le comportement de la mémoire de projet de Qwen Chat. Garde un fil de conversation continu, ou rappelle en une phrase où tu en es (« on est à la tâche T18.3 »).

### 0.2 Pour Qwen Chat

*(mode d'emploi que Qwen Chat doit suivre — cette sous-section reste dans les fichiers du projet)*

Tu es l'**orchestrateur et rédacteur de code** de ce projet. Ton rôle se limite à deux choses :

1. **Dire quoi faire ensuite.** Consulte la table de suivi (V2 §9.0) : identifie la première tâche non cochée (⬜), présente-la (ID, phase, fichier) et résume sa spécification à partir de la section 9. Si l'état d'avancement ne t'a pas été communiqué dans cette conversation, demande-le plutôt que de le deviner.
2. **Composer le code.** Assemble le prompt de génération de code avec le gabarit de la V2 §8 : contexte permanent (V1 §2, V1 §6, V1 §7, V2 §3.2, V2 §7 — recopiés intégralement, ce sont des contrats) + contexte de la phase (V2 §9) + la tâche précise. Ensuite exécute ce prompt et livre le(s) code(s) entier(s), prêt(s) à copier-coller.

**Ce que tu ne dois jamais faire :**

- Écrire toi-même l'implémentation.
- Modifier une signature ou un contrat de la V1 §7 ou de la V2 §7 sans demande explicite de l'utilisateur.
- Sauter une tâche ou réordonner les phases. L'ordre existe pour que chaque famille de méthodes soit validée contre R avant d'être exposée à l'IA (phase 27).
- Répondre « où en est-on » sans t'appuyer sur la table 9.0.
- Proposer une dépendance qui n'est pas listée en V2 §4.3 sans la signaler explicitement comme « à valider ».
- Accepter d'ajouter une fonctionnalité listée en V2 §1.3 (hors périmètre). Si l'utilisateur la demande, rappelle que c'est hors périmètre V2 et demande confirmation.

---

## 1. Contexte et périmètre de la V2

### 1.1 État de départ

La V1 est terminée : import CSV/XLSX/SAV, profiling, dictionnaire de données, moteur statistique (χ², t, ANOVA, OLS, logistique, MNLogit, Firth via pont R), `AnalysisResult`, génération de code et de rapports HTML/Markdown, diagnostics V1 (VIF, séparation, influence, taille d'échantillon), couche IA (tool calling), Analysis Planner, agent de correction, interface Streamlit, reproductibilité (sauvegarde/rejeu), sandbox de sécurité.

### 1.2 Objectif de la V2

Étendre la couverture statistique du Statistical IDE aux méthodes que **l'écosystème Python et R permet d'intégrer sans les réécrire**, en conservant la règle V1 : *l'IA propose, le moteur calcule, le validateur contrôle, l'utilisateur décide.* L'objectif n'est **pas** la parité avec SPSS/SAS/Stata, mais une couverture solide pour des chercheurs, étudiants et statisticiens individuels, avec diagnostics, code reproductible et validation contre R.

Contraintes inchangées : développeur solo, budget quasi nul, machine CPU-only (~12 Go RAM, sans GPU), Linux Mint, local-first.

### 1.3 Périmètre : inclus, partiel, exclu

Correspondance avec la liste de 25 catégories de fonctionnalités :

| # | Catégorie | Décision V2 | Phase(s) |
|---|-----------|-------------|----------|
| 1 | Gestion et manipulation des données | **Inclus** (sauf NoSQL, ODBC/JDBC, cloud, éditeur complet) | 15, 27 |
| 2 | Descriptif et exploratoire | **Inclus** | 16 |
| 3 | Inférence statistique | **Inclus** | 17 |
| 4 | Modèles linéaires et extensions | **Inclus** (modèles non linéaires/GAM : partiel, splines seulement) | 18, 19 |
| 5 | Analyse multivariée | **Inclus** (autoencodeurs exclus ; k-medoids reporté) | 20 |
| 6 | Séries temporelles et économétrie | **Inclus** (Tobit/Heckman/GMM dynamique : via R, optionnel) | 22 |
| 7 | Enquêtes et données complexes | **Inclus via R** (`survey`) ; petits domaines exclus | 23 |
| 8 | Survie et fiabilité | **Inclus** (frailty via R) | 21 |
| 9 | Méta-analyse | **Inclus** (Python + R) | 23 |
| 10 | Contrôle qualité et plans d'expérience | **Optionnel**, petit périmètre | 28 |
| 11 | Apprentissage statistique et ML | **Inclus** (ML classique, pas de deep learning) | 25 |
| 12 | Méthodes bayésiennes | **Optionnel**, petits modèles | 28 |
| 13 | Inférence causale | **Inclus partiel** (TMLE, contrôles synthétiques reportés) | 24 |
| 14 | Simulation et calcul statistique | **Partiel** (bootstrap, permutations, puissance par simulation) | 17 |
| 15 | Graphiques et visualisation | **Inclus** (statique + Plotly interactif ; 3D/cartes exclus) | 16, 26 |
| 16 | Reporting et communication | **Inclus** (HTML, Word, LaTeX, PDF, notebooks) | 26 |
| 17 | Programmation et reproductibilité | **Déjà couvert V1** + export de scripts/notebooks | 11 (V1), 26 |
| 18 | Interface utilisateur | **Partiel** (Streamlit : éditeur de données, projets, aide) | 27 |
| 19 | Collaboration et entreprise | **Exclu** | — |
| 20 | Performance et architecture | **Partiel** (chunking, Parquet ; pas de GPU/Spark) | 15, 29 |
| 21 | Extensibilité et intégration | **Partiel** (registre interne de méthodes ; pas de marché de plugins) | 15 |
| 22 | Documentation, formation, support | **Partiel** (documentation générée ; pas de certification) | 29 |
| 23 | Conformité réglementaire | **Exclu** (la traçabilité de la V1/V2 aide mais ne vaut pas validation GxP) | — |
| 24 | Fonctionnalités spécialisées par domaine | **Exclu** sauf ce qui est déjà couvert ailleurs | — |
| 25 | Types de données spéciaux | **Couvert** par les phases 19, 21, 22 ; texte/image/graphes exclus | — |

**Hors périmètre V2 (décision explicite — ne pas implémenter, ne pas proposer) :**

- Multi-utilisateurs, rôles, SSO, collaboration temps réel, HIPAA, 21 CFR Part 11, signatures électroniques, conformité FDA/EMA/CDISC/GxP.
- Calcul GPU, calcul distribué, Spark, Dask, GraphQL, marché de plugins, certifications, forum et support.
- Deep learning, autoencodeurs, NLP, analyse d'images et de signaux, ERGM et analyse de réseaux sociaux, SIG et géostatistique, petits domaines, modèles pour données fonctionnelles/compositionnelles/circulaires.
- Import NoSQL, ODBC/JDBC et connecteurs cloud (S3, GCS, etc.).
- Éditeur de données complet de type tableur (on se limite à `st.data_editor`, V2 §9 phase 27).
- MLOps (déploiement de modèles, monitoring de dérive en production).

Si l'un de ces points devient nécessaire plus tard, il fera l'objet d'une spécification dédiée (V3), pas d'un ajout silencieux.

---

## 2. Règle architecturale (rappel) et règles ajoutées

### 2.1 Rappel — à coller dans CHAQUE session (identique à la V1 §2)

```
L'IA PROPOSE  →  LE MOTEUR CALCULE  →  LE VALIDATEUR CONTRÔLE  →  L'UTILISATEUR DÉCIDE
```

Toutes les règles dérivées de la V1 §2 restent en vigueur (aucun LLM ne calcule, aucun import de `app.agent.*` dans `app/data`, `app/statistics`, `app/diagnostics`, `app/results`, le module agent ne voit jamais de DataFrame brut, etc.).

### 2.2 Règles ajoutées par la V2

- **Aucun nouveau module de calcul** (`app/statistics/*`, `app/multivariate/*`, `app/survival/*`, `app/timeseries/*`, `app/causal/*`, `app/ml/*`, `app/survey/*`, `app/meta/*`) ne doit importer `app.agent.*`, `app.planner.*` ou `app.ui.*`.
- **Les données des utilisateurs ne quittent jamais la machine.** Le pont R s'exécute en local ; aucun appel réseau dans les modules de calcul (T4 hérité).
- **Le LLM ne choisit jamais un paramètre numérique de calcul** (niveau de signification, nombre de clusters, ordre d'un ARIMA, etc.) sans que ce paramètre soit affiché à l'utilisateur dans le plan d'analyse et validé (T14 hérité). Les valeurs par défaut sont définies dans le code déterministe, pas dans un prompt.
- **Tout choix automatique du moteur** (sélection d'un nombre de clusters, d'un ordre ARIMA, d'une pénalité lasso) doit être consigné dans `execution_log` et dans `model_metrics`.
- **Les méthodes sans p-values** (régression régularisée, ML) renvoient `p_values = {}` et ajoutent un avertissement explicite dans `warnings` : le moteur ne fabrique jamais une p-value que la méthode ne fournit pas.

---

## 3. Contraintes techniques

### 3.1 Contraintes héritées (inchangées)

T1 à T15 de la V1 §3 restent en vigueur, en particulier : T1 (code simple), T3 (CPU-only, pandas par défaut), T4 (aucun appel réseau dans les modules de calcul), T5 (typage strict), T7 (ne pas réinventer les maths), T9 (LLM provider-agnostic), T12 (validation contre R), T13 (code reproductible retourné), T14 (confirmation humaine), T15 (pas d'`eval`/`exec` sur du texte généré sans sandbox).

### 3.2 Nouvelles contraintes V2 (à coller dans CHAQUE session)

| # | Contrainte | Directive pour le code généré |
|---|-----------|-------------------------------|
| T16 | Une méthode = un `MethodSpec` | Toute méthode exposée à l'utilisateur est enregistrée dans le registre (V2 §7.1). Pas de méthode « cachée » appelée directement par l'UI ou l'agent. |
| T17 | Contrats V1 figés | `AnalysisResult`, `Variable`, `DataDictionary`, `DiagnosticFinding`, `Severity`, `StatisticalEngineProvider`, `LLMProvider`, `AnalysisPlan` ne changent pas. Les besoins nouveaux passent par les conventions de la V2 §6.2 (clés de `model_metrics`, tables) ou par de **nouveaux** types. |
| T18 | Imports paresseux des dépendances lourdes | `lifelines`, `linearmodels`, `arch`, `prince`, `factor_analyzer`, `umap`, `shap`, `pymc`, `weasyprint`, etc. s'importent **dans** la fonction qui les utilise, jamais au niveau du module. L'application et les tests de base doivent démarrer sans elles. |
| T19 | Dépendances optionnelles par « extras » | Les dépendances des phases 20 à 28 sont ajoutées avec `uv add --optional <extra> <paquet>`. Une méthode dont l'extra est absent lève une exception claire en français (`DependencyMissingError`, V2 §7.4), jamais un `ImportError` brut. |
| T20 | R uniquement par le pont | Tout appel à R passe par `app/r_bridge`. Pas de `rpy2` (dépendance fragile, hors liste). Dégradation propre si R est absent (déjà exigée en V1 Phase 6). |
| T21 | Tolérances de validation documentées | Chaque test de validation contre R déclare sa tolérance et la justifie en commentaire (V2 §6.3). Les méthodes stochastiques utilisent une graine fixe et comparent des **propriétés** (pas des valeurs exactes) quand l'algorithme diffère de celui de R. |
| T22 | Budget de ressources | Aucune fonction ne doit dépasser ~4 Go de RAM sur un dataset de 1 million de lignes × 20 colonnes. Pas de copie inutile du DataFrame ; `copy=False` / vues quand c'est sûr. Si une méthode est inadaptée à cette taille (ex. DBSCAN, SHAP), elle doit **échantillonner** et le signaler dans `warnings`. |
| T23 | Résultats sérialisables | Tout contenu de `AnalysisResult` doit être sérialisable en JSON (`to_json()`) : convertir `numpy.float64`, `NaN`, `Timestamp`, etc. en types natifs. Les figures ne sont jamais stockées dans `AnalysisResult`. |
| T24 | Aucune « boîte noire » sans avertissement | Toute méthode exploratoire ou prédictive (clustering, t-SNE/UMAP, ML) doit ajouter dans `warnings` ce qu'elle ne permet pas de conclure (pas de causalité, distances non interprétables, etc.). |

---

## 4. Stack technique et dépendances par phase

### 4.1 Prérequis système

Ceux de la V1 §4.1 (Python 3.11+, `uv`, R depuis la Phase 6). La V2 ajoute :

```bash
# Packages R utilisés par le pont (installer uniquement au moment de la phase concernée)
Rscript -e 'install.packages(c("lme4","geepack"), repos="https://cloud.r-project.org")'                  # phase 19
Rscript -e 'install.packages(c("lavaan","psych","FactoMineR"), repos="https://cloud.r-project.org")'      # phase 20
Rscript -e 'install.packages(c("survival","coxme"), repos="https://cloud.r-project.org")'                 # phase 21
Rscript -e 'install.packages(c("forecast","urca","tseries","plm","AER","sampleSelection"), repos="https://cloud.r-project.org")'  # phase 22
Rscript -e 'install.packages(c("survey","metafor"), repos="https://cloud.r-project.org")'                 # phase 23
Rscript -e 'install.packages(c("MatchIt","rdrobust"), repos="https://cloud.r-project.org")'               # phase 24
Rscript -e 'install.packages(c("glmnet","MASS"), repos="https://cloud.r-project.org")'                    # phase 18 (validation)
Rscript -e 'install.packages(c("qcc"), repos="https://cloud.r-project.org")'                              # phase 28 (optionnel)

# Phase 26 uniquement, pour l'export PDF via WeasyPrint
sudo apt install -y libpango-1.0-0 libpangoft2-1.0-0
```

> Les packages R servent soit de **référence de validation** (T12), soit de **moteur** via le pont (enquêtes, méta-analyse, SEM, frailty). Pour chaque phase, la tâche de tests indique lequel.

### 4.2 Extras optionnels (T19)

Les extras sont déclarés dans `pyproject.toml` via `uv add --optional <extra> ...`. La CI V1 exécute déjà `uv sync --all-extras --dev` : les tests des extras tournent donc en CI.

| Extra | Contenu | Phase |
|-------|---------|-------|
| `multivariate` | factor-analyzer, prince | 20 |
| `embedding` | umap-learn *(tire `numba`, lourd : extra séparé, jamais requis)* | 20 |
| `survival` | lifelines | 21 |
| `timeseries` | linearmodels, arch | 22 |
| `causal` | rdrobust, networkx | 24 |
| `ml` | shap, imbalanced-learn | 25 |
| `reporting` | python-docx, nbformat, weasyprint | 26 |
| `bayes` | pymc, arviz | 28 |
| `design` | pyDOE3 | 28 |

### 4.3 Dépendances par phase (commandes exactes)

*Ne jamais ajouter une dépendance d'une phase ultérieure en avance. `uv add` verrouille les versions dans `uv.lock` ; ne pas fixer de numéro de version à la main sauf raison documentée. Avant d'utiliser une bibliothèque, on doit vérifier l'API réelle de la version installée (`uv run python -c "import X; help(X)"`) : les noms de classes et de fonctions cités dans ce document sont indicatifs.*

| Phase | Commande |
|-------|----------|
| 15 — Socle V2 + Data Engine V2 | `uv add openpyxl pyarrow lxml sqlalchemy` *(openpyxl seulement s'il est absent)* |
| 16 — Descriptif et EDA | *(aucune nouvelle dépendance)* |
| 17 — Inférence V2 | `uv add pingouin` *(vérifier la compatibilité de versions avec statsmodels/pandas avant de valider)* |
| 18 — Régression V2 | `uv add scikit-learn` |
| 19 — Modèles mixtes et longitudinal | *(aucune nouvelle dépendance Python ; packages R `lme4`, `geepack`)* |
| 20 — Multivarié | `uv add --optional multivariate factor-analyzer prince` ; `uv add --optional embedding umap-learn` *(optionnel)* |
| 21 — Survie et fiabilité | `uv add --optional survival lifelines` |
| 22 — Séries temporelles, panel, économétrie | `uv add --optional timeseries linearmodels arch` |
| 23 — Enquêtes et méta-analyse | *(aucune nouvelle dépendance Python ; packages R `survey`, `metafor`)* |
| 24 — Causalité | `uv add --optional causal rdrobust networkx` |
| 25 — ML et interprétabilité | `uv add --optional ml shap imbalanced-learn` |
| 26 — Reporting étendu | `uv add --optional reporting python-docx nbformat weasyprint` |
| 27 — Planner V2, agent, UI V2 | *(aucune nouvelle dépendance)* |
| 28 — Optionnel : bayésien, qualité, plans d'expérience | `uv add --optional bayes pymc arviz` ; `uv add --optional design pyDOE3` |
| 29 — Stabilisation | *(aucune nouvelle dépendance)* |

### 4.4 `pyproject.toml` — ajouts

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "r_required: nécessite R et ses packages (ignoré en CI sans R)",
    "extra: nécessite un extra optionnel installé",
    "slow: test de plus de 5 secondes",
]

[[tool.mypy.overrides]]
module = [
    "lifelines.*", "linearmodels.*", "arch.*", "prince.*", "factor_analyzer.*",
    "pingouin.*", "shap.*", "rdrobust.*", "umap.*", "weasyprint.*", "pymc.*", "arviz.*",
]
ignore_missing_imports = true
```

### 4.5 CI — ajout d'un job avec R (optionnel, planifié)

Le job principal de la V1 (sans R) reste inchangé. Ajouter un second workflow `.github/workflows/ci-r.yml`, déclenché manuellement ou chaque semaine :

```yaml
name: CI-R
on:
  workflow_dispatch:
  schedule:
    - cron: "0 3 * * 1"
jobs:
  test-r:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: r-lib/actions/setup-r@v2
      - uses: astral-sh/setup-uv@v3
      - run: Rscript -e 'install.packages(c("jsonlite","logistf","lme4","geepack","survival","survey","metafor","MatchIt"))'
      - run: uv sync --all-extras --dev
      - run: uv run pytest -m "r_required or not r_required"
```

*(Vérifier les versions des actions au moment de l'installation ; la liste de packages R s'ajuste aux phases terminées.)*

---

## 5. Organisation du projet (arborescence cible V2)

*(Ajouts à l'arborescence V1 §5. Ne créer un dossier ou un fichier que lorsque sa première tâche arrive.)*

```
statistical-ai/
├── app/
│   ├── data/
│   │   ├── exporters.py             # export CSV/XLSX/Parquet/Stata/LaTeX/HTML (Phase 15)
│   │   ├── transform_log.py         # TransformStep, TransformLog (Phase 15)
│   │   ├── transforms.py            # recodage, calcul, standardisation, dates (Phase 15)
│   │   ├── missing.py               # détection, imputation simple/multiple (Phase 15)
│   │   ├── manipulation.py          # fusion, jointures, pivot, dédoublonnage (Phase 15)
│   │   └── large.py                 # chunking, downcast, échantillonnage (Phase 15)
│   ├── statistics/
│   │   ├── registry.py              # MethodSpec, register, get_method, list_methods (Phase 15)
│   │   ├── correlation.py           # corrélations (Phase 16)
│   │   ├── normality_outliers.py    # normalité, outliers (Phase 16)
│   │   ├── tests_parametric.py      # t pairé, Welch, proportions, variances (Phase 17)
│   │   ├── tests_nonparam.py        # Wilcoxon, Mann-Whitney, Kruskal, Friedman (Phase 17)
│   │   ├── tests_anova.py           # ANCOVA, MANOVA, post-hoc (Phase 17)
│   │   ├── effect_sizes.py          # tailles d'effet et IC (Phase 17)
│   │   ├── power.py                 # puissance, taille d'échantillon (Phase 17)
│   │   ├── multiple_testing.py      # Bonferroni, Holm, FDR, TOST (Phase 17)
│   │   ├── resampling.py            # bootstrap, permutation (Phase 17)
│   │   ├── regression_glm.py        # probit, Poisson, NB, Gamma, ordinale (Phase 18)
│   │   ├── regression_regularized.py# ridge, lasso, elastic net (Phase 18)
│   │   ├── regression_robust.py     # robuste, polynomiale, splines (Phase 18)
│   │   └── mixed.py                 # MixedLM, GEE, GLMM via R (Phase 19)
│   ├── multivariate/                # ACP, AFE, AFC/ACM, clustering, LDA/QDA (Phase 20)
│   ├── survival/                    # KM, Cox, AFT, risques concurrents, fiabilité (Phase 21)
│   ├── timeseries/                  # décomposition, ARIMA, VAR, GARCH, prévision (Phase 22)
│   ├── econometrics/                # panel, IV, Tobit/Heckman via R (Phase 22)
│   ├── survey/                      # SurveyDesign, estimations pondérées via R (Phase 23)
│   ├── meta/                        # méta-analyse (Phase 23)
│   ├── causal/                      # propension, appariement, IPTW, DiD, RDD (Phase 24)
│   ├── ml/                          # pipelines, modèles, métriques, interprétabilité (Phase 25)
│   ├── quality/                     # cartes de contrôle, plans d'expérience (Phase 28, optionnel)
│   ├── bayes/                       # modèles bayésiens simples (Phase 28, optionnel)
│   ├── diagnostics/
│   │   ├── checks_inference.py      # (Phase 17)
│   │   ├── checks_regression.py     # (Phase 18)
│   │   └── checks_<famille>.py      # un fichier par famille, au fil des phases
│   ├── reports/
│   │   ├── plots_eda.py             # graphiques exploratoires (Phase 16)
│   │   ├── publication.py           # PublicationTable, tableaux de publication (Phase 26)
│   │   └── exporters.py             # Word, LaTeX, PDF, notebook (Phase 26)
│   ├── r_bridge/
│   │   └── scripts/                 # un script .R par méthode via R (phases 19 à 24)
│   └── errors.py                    # DependencyMissingError, etc. (Phase 15)
├── tests/
│   ├── benchmarks/                  # jeux de données + résultats R attendus (JSON, V2 §7.5)
│   └── test_<famille>/              # un dossier de tests par famille
└── docs/
    └── adr/ADR-003-*.md, ADR-004-*.md, ADR-005-*.md
```

---

## 6. Conventions de code (compléments à la V1 §6)

Toutes les conventions de la V1 §6 restent en vigueur (anglais pour les identifiants, français pour les docstrings et messages, typage strict, Google docstrings, un fichier de test par module, Conventional Commits, pas de code mort).

### 6.1 Identifiants de méthodes

`method_id` : `snake_case`, anglais, stable, jamais renommé après publication (il est archivé dans les analyses sauvegardées par la Phase 11 V1). Exemples : `mann_whitney`, `kruskal_wallis`, `poisson_glm`, `cox_ph`, `arima`, `iv_2sls`.

### 6.2 Utilisation de `AnalysisResult` pour les nouvelles méthodes (contrat V1 inchangé, T17)

- `estimates` : dictionnaire `nom → valeur` (float natif). Pour un test : `estimates["statistic"]`. Pour un modèle : un coefficient par terme.
- `p_values` : `{terme: p}` ; pour un test global : `p_values["overall"]`. Vide `{}` si la méthode n'en fournit pas (ET `warnings` l'explique).
- `model_metrics` : dictionnaire libre, clés en `snake_case`, **toujours** avec `"method_family"` (valeur du `MethodSpec.family`). Autres clés usuelles : `"effect_size"`, `"adjusted_p_values"`, `"random_state"`, `"n_obs_used"`, `"n_obs_dropped"`.
- Résultats tabulaires (paires de post-hoc, charges factorielles, prévisions, courbes de survie, etc.) : `model_metrics["tables"][nom_table]` = **liste de dictionnaires** (un par ligne), JSON-sérialisable (T23).
- `preprocessing` : inclure la gestion des manquants (« suppression des lignes incomplètes : n lignes retirées ») et le résumé du `TransformLog` s'il y en a un.
- `software_versions` : versions des bibliothèques Python utilisées ; version de R et des packages R si le pont a été appelé.
- `generated_code` : script autonome (T13). Pour y parvenir, le module `app/results/codegen.py` doit passer à un **dispatch par dictionnaire** (`register_codegen(method_id, fn)`) sans changer la signature publique de `generate_reproducible_code`.

### 6.3 Tests de validation contre R (T12, T21)

- Les valeurs de référence R sont stockées dans `tests/benchmarks/<method_id>.json` (schéma : V2 §7.5), jamais recalculées à l'exécution des tests.
- Le script R qui a produit la référence est stocké à côté (`tests/benchmarks/<method_id>.R`) pour pouvoir être rejoué.
- Tolérances recommandées (à ajuster et à justifier en commentaire) :
  - formules fermées (tests classiques, OLS) : `rel=1e-6` ;
  - optimisation itérative (GLM, mixtes, Cox, ARIMA) : `rel=1e-4` ;
  - méthodes stochastiques (k-means, bootstrap, MCMC) : graine fixe, comparaison de **propriétés** (nombre de clusters, ordre des modalités, intervalle contenant la valeur de R), pas d'égalité exacte.

---

## 7. Contrats de données et interfaces nouveaux

*(à coller à Qwen dans CHAQUE session, avec V1 §7 — ne jamais laisser Qwen les modifier sans validation explicite de ta part)*

### 7.1 `app/statistics/registry.py` — registre de méthodes

```python
from dataclasses import dataclass, field
from typing import Callable

from app.results.models import AnalysisResult


@dataclass(frozen=True)
class MethodSpec:
    """Description déclarative d'une méthode statistique exposée à l'utilisateur."""

    method_id: str                    # ex. "mann_whitney" (snake_case, stable)
    label_fr: str                     # libellé affiché
    family: str                       # "descriptive" | "inference" | "regression" | "mixed" |
                                      # "multivariate" | "survival" | "timeseries" | "panel" |
                                      # "survey" | "meta" | "causal" | "ml" | "bayes" | "quality"
    engine: str                       # "python" | "r"
    outcome_types: list[str]          # semantic_type compatibles ("continuous", "binary", ...)
    predictor_types: list[str]
    min_sample_size: int
    assumptions: list[str]            # hypothèses à vérifier (texte français)
    reference_impl: str               # ex. "scipy.stats.mannwhitneyu"
    r_reference: str | None           # ex. "stats::wilcox.test"
    phase: int
    extra: str | None = None          # nom de l'extra uv requis, ou None
    default_params: dict = field(default_factory=dict)


def register_method(spec: MethodSpec, func: Callable[..., AnalysisResult]) -> None: ...
def get_method(method_id: str) -> tuple[MethodSpec, Callable[..., AnalysisResult]]: ...
def list_methods(family: str | None = None) -> list[MethodSpec]: ...
```

Règles : un `method_id` ne peut être enregistré qu'une fois (lever `ValueError` sinon) ; l'enregistrement se fait à l'import du module de la méthode ; `list_methods()` ne doit **jamais** importer une dépendance optionnelle (T18).

### 7.2 `app/data/transform_log.py` — traçabilité des transformations

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TransformStep:
    """Une opération appliquée à un DataFrame, rejouable."""

    step_id: str
    operation: str                  # ex. "recode", "standardize", "impute_median"
    params: dict
    columns_in: list[str]
    columns_out: list[str]
    n_rows_before: int
    n_rows_after: int
    code: str                       # extrait Python reproduisant l'opération
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TransformLog:
    """Journal chronologique des transformations d'un dataset."""

    steps: list[TransformStep] = field(default_factory=list)

    def add(self, step: TransformStep) -> None: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, json_str: str) -> "TransformLog": ...
    def to_python_script(self) -> str: ...   # script autonome rejouant toutes les étapes
```

### 7.3 `app/survey/design.py` — plan de sondage

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SurveyDesign:
    """Plan de sondage déclaré par l'utilisateur (jamais déduit par un LLM)."""

    weight: str | None = None       # colonne des poids
    strata: str | None = None
    cluster: str | None = None      # unité primaire d'échantillonnage (UPE)
    fpc: str | None = None          # correction de population finie
    nest: bool = False
    domain: str | None = None       # colonne définissant les sous-populations
```

### 7.4 `app/errors.py` — erreurs métier

```python
class StatisticalIDEError(Exception):
    """Base de toutes les erreurs métier ; le message est en français."""


class DependencyMissingError(StatisticalIDEError):
    """Un extra optionnel ou R est requis mais absent. Le message indique
    la commande d'installation exacte (ex. « uv add --optional survival lifelines »)."""


class InvalidMethodInputError(StatisticalIDEError):
    """Les données ou paramètres ne conviennent pas à la méthode (message clair, en français)."""
```

### 7.5 Schéma des fichiers de référence R (`tests/benchmarks/<method_id>.json`)

```json
{
  "method_id": "mann_whitney",
  "dataset": "tests/benchmarks/data/mann_whitney_demo.csv",
  "r_version": "4.x.y",
  "r_function": "stats::wilcox.test",
  "r_script": "tests/benchmarks/mann_whitney.R",
  "params": {"outcome": "score", "group": "traitement"},
  "expected": {"statistic": 123.0, "p_value": 0.0421},
  "rtol": 1e-6,
  "tolerance_rationale": "formule exacte, pas d'itération"
}
```

### 7.6 Catalogue des codes de diagnostic ajoutés

*(Même `DiagnosticFinding` et même `Severity` qu'en V1. Les codes sont déterministes ; aucun n'est produit par un LLM. Les seuils sont documentés en commentaire avec leur source.)*

| Code | Sévérité par défaut | Détecte | Phase |
|------|--------------------|---------|-------|
| `HIGH_MISSING_RATE` | WARNING | variable avec fort taux de valeurs manquantes | 15 |
| `CONSTANT_VARIABLE` | CERTAIN_ISSUE | variable constante utilisée comme prédicteur | 15 |
| `NON_NORMAL_DISTRIBUTION` | WARNING | écart à la normalité (échantillon, résidus) | 16, 18 |
| `OUTLIERS_DETECTED` | INFORMATION | valeurs extrêmes | 16 |
| `UNEQUAL_VARIANCES` | PROBABLE_ISSUE | hétérogénéité des variances entre groupes | 17 |
| `LOW_EXPECTED_COUNTS` | PROBABLE_ISSUE | effectifs théoriques < 5 (χ²) | 17 |
| `SMALL_GROUP_SIZE` | WARNING | groupe trop petit | 17 |
| `HETEROSCEDASTICITY` | PROBABLE_ISSUE | variance non constante des résidus | 18 |
| `AUTOCORRELATED_RESIDUALS` | PROBABLE_ISSUE | autocorrélation des résidus | 18, 22 |
| `OVERDISPERSION` | PROBABLE_ISSUE | surdispersion (Poisson) | 18 |
| `NONCONVERGENCE` | CERTAIN_ISSUE | l'optimiseur n'a pas convergé | 18, 19, 21, 22 |
| `SINGULAR_FIT` | PROBABLE_ISSUE | effet aléatoire dégénéré | 19 |
| `LOW_KMO` | PROBABLE_ISSUE | données inadaptées à l'analyse factorielle | 20 |
| `PROPORTIONAL_HAZARDS_VIOLATED` | PROBABLE_ISSUE | hypothèse de Cox non respectée | 21 |
| `NON_STATIONARY_SERIES` | PROBABLE_ISSUE | série non stationnaire | 22 |
| `WEAK_INSTRUMENT` | PROBABLE_ISSUE | instrument faible | 22 |
| `HIGH_DESIGN_EFFECT` | WARNING | effet de plan de sondage élevé | 23 |
| `HIGH_HETEROGENEITY` | WARNING | I² élevé en méta-analyse | 23 |
| `POOR_COVARIATE_BALANCE` | PROBABLE_ISSUE | déséquilibre résiduel après appariement/pondération | 24 |
| `LEAKAGE_SUSPECTED` | PROBABLE_ISSUE | fuite d'information probable (ML) | 25 |
| `OVERFITTING_SUSPECTED` | WARNING | écart important entre apprentissage et validation | 25 |
| `CLASS_IMBALANCE` | WARNING | classes très déséquilibrées | 25 |

---

## 8. Gabarit de prompt pour Qwen

Utilise ce gabarit pour **chaque** tâche. Ne saute jamais la partie « Contexte permanent ».

```
### Contexte permanent (colle ici : V1 §2, V1 §6, V1 §7, V2 §3.2, V2 §7)

### Contexte de phase (colle ici la sous-section de la phase en cours, V2 §9)

### Tâche

Implémente uniquement ce qui suit, rien de plus :

Fichier : <chemin exact, ex. app/statistics/tests_nonparam.py>

Signature à respecter exactement :
def run_mann_whitney(df: pd.DataFrame, outcome: str, group: str) -> AnalysisResult:
    """..."""

Spécification :
- <détails fonctionnels tirés de la section 9>

Contraintes :
- Respecte exactement la signature ci-dessus, ne change aucun contrat de V1 §7 ni de V2 §7
- N'utilise que les dépendances déjà installées à cette phase (V2 §4.3) ; imports paresseux pour les extras (T18)
- Respecte la Définition de Done commune (V2 §9.1) : fonction + enregistrement dans le registre
  + gabarit de code reproductible + test nominal + test cas limite + test de validation contre R
- Fournis aussi les tests pytest correspondants dans <chemin du fichier de test>
- Si tu penses qu'une dépendance ou un contrat doit changer, signale-le au lieu de le faire silencieusement
```

**Exemple concret rempli (Phase 17, une tâche) :**

```
### Tâche

Fichier : app/statistics/tests_nonparam.py

Signature à respecter exactement :
def run_mann_whitney(df: pd.DataFrame, outcome: str, group: str) -> AnalysisResult:
    """..."""

Spécification :
- Utilise scipy.stats.mannwhitneyu (use_continuity=True, alternative="two-sided")
- group doit avoir exactement 2 modalités (lever InvalidMethodInputError sinon, message en français)
- estimates["statistic"] = U ; p_values["overall"] = p ; model_metrics["effect_size"] = {"rank_biserial": ...}
- Supprime les lignes incomplètes sur (outcome, group) et indique le nombre retiré dans preprocessing
- Ajoute un DiagnosticFinding SMALL_GROUP_SIZE si un groupe a moins de 10 observations

Contraintes :
- Enregistre la méthode dans le registre avec un MethodSpec (family="inference", engine="python")
- Fournis le gabarit de code reproductible via register_codegen("mann_whitney", ...)
- Tests dans tests/test_statistics/test_tests_nonparam.py : cas nominal, cas limite (groupe à une modalité),
  validation contre R (tests/benchmarks/mann_whitney.json, wilcox.test, rel=1e-6)
```

---

## 9. Guide d'implémentation par phase

### 9.1 Définition de Done commune à toute nouvelle méthode statistique

Une tâche qui ajoute une méthode **n'est terminée** que si les six points suivants sont vrais :

1. **Fonction** : signature complète, type hints, docstring française, retourne un `AnalysisResult` conforme à V2 §6.2.
2. **Registre** : `MethodSpec` enregistré (V2 §7.1) avec hypothèses, référence Python et référence R.
3. **Code reproductible** : gabarit enregistré via `register_codegen` ; le script généré s'exécute seul (T13).
4. **Diagnostics** : les diagnostics pertinents sont appelés et leurs `DiagnosticFinding` placés dans `result.diagnostics`.
5. **Tests** : cas nominal, cas limite, et **validation contre R** (V2 §6.3) — ou, pour une méthode sans équivalent R raisonnable, une validation contre une référence indépendante documentée (valeur publiée, formule analytique) justifiée en commentaire.
6. **Qualité** : `uv run ruff check .`, `uv run mypy app/` et `uv run pytest` passent, sans dépendance LLM (V1 §2).

### 9.2 Jalons

| Jalon | Après la phase | Ce que tu peux publier |
|-------|---------------|------------------------|
| **V2-alpha** | 19 | Statistiques classiques complètes : données, EDA, inférence, GLM, mixtes |
| **V2-beta** | 23 | + multivarié, survie, séries temporelles, panel, enquêtes, méta-analyse |
| **V2-RC** | 27 | + causalité, ML, reporting, planner et UI étendus |
| **V2.0.0** | 29 | Version stabilisée et documentée |

À chaque jalon : `uv run pytest` vert en local et en CI, rapport de validation contre R à jour, `CHANGELOG.md` mis à jour, tag Git (`v1.1.0`, `v1.2.0`, `v1.3.0`, `v2.0.0`).

### 9.0 Suivi d'avancement

*(Source de vérité pour répondre à « où en est-on » / « par quoi on commence ». Statut mis à jour manuellement par l'humain — V2 §0.1. Chaque ID correspond, dans l'ordre, aux tâches numérotées de la phase.)*

| ID | Phase | Tâche (résumé) | Fichier principal | Statut |
|----|-------|----------------|-------------------|--------|
| T15.1 | 15 — Socle V2 + Data Engine V2 | ADR-003 (registre), ADR-004 (extras), ADR-005 (R pour enquêtes/méta/SEM) — **(humain)** | docs/adr/ADR-003-*.md à 005-*.md | ✅ |
| T15.2 | 15 — Socle V2 + Data Engine V2 | Erreurs métier + MethodSpec/registre + enregistrement des méthodes V1 + codegen par dispatch | app/errors.py, app/statistics/registry.py | ✅ |
| T15.3 | 15 — Socle V2 + Data Engine V2 | Importeurs étendus (JSON, Parquet, Stata, SAS, XML, SQL/SQLite) | app/data/importers.py | ✅ |
| T15.4 | 15 — Socle V2 + Data Engine V2 | Exporteurs (CSV, XLSX, Parquet, Stata, LaTeX, HTML) | app/data/exporters.py | ✅ |
| T15.5 | 15 — Socle V2 + Data Engine V2 | TransformStep / TransformLog | app/data/transform_log.py | ✅ |
| T15.6 | 15 — Socle V2 + Data Engine V2 | Transformations (recodage, calcul, standardisation, centrage, discrétisation, dates) | app/data/transforms.py | ✅ |
| T15.7 | 15 — Socle V2 + Data Engine V2 | Valeurs manquantes (détection, recodage, imputation simple et multiple) | app/data/missing.py | ✅ |
| T15.8 | 15 — Socle V2 + Data Engine V2 | Manipulation (fusion, jointures, agrégation, pivot, dédoublonnage, échantillonnage) | app/data/manipulation.py | ✅ |
| T15.9 | 15 — Socle V2 + Data Engine V2 | Gros volumes (chunking, downcast, échantillonnage) | app/data/large.py | ✅ |
| T15.10 | 15 — Socle V2 + Data Engine V2 | Dictionnaire enrichi (unités, niveau de mesure, rôles, poids) | app/data/dictionary_tools.py | ✅ |
| T15.11 | 15 — Socle V2 + Data Engine V2 | Tests + fixtures V2 + validation de phase | tests/test_data/ | ✅ |
| T16.1 | 16 — Descriptif et EDA | Fréquences, résumés descriptifs étendus | app/statistics/descriptive.py | ✅ |
| T16.2 | 16 — Descriptif et EDA | Corrélations (Pearson, Spearman, Kendall, matrices) | app/statistics/correlation.py | ✅ |
| T16.3 | 16 — Descriptif et EDA | Normalité et détection d'outliers | app/statistics/normality_outliers.py | ✅ |
| T16.4 | 16 — Descriptif et EDA | Graphiques exploratoires | app/reports/plots_eda.py | ✅ |
| T16.5 | 16 — Descriptif et EDA | Rapport EDA automatique | app/reports/generator.py | ✅ |
| T16.6 | 16 — Descriptif et EDA | Tests + validation contre R | tests/test_statistics/ | ✅ |
| T17.1 | 17 — Inférence V2 | t pairé, Welch, proportions, Fisher, variances | app/statistics/tests_parametric.py | ✅ |
| T17.2 | 17 — Inférence V2 | Tests non paramétriques | app/statistics/tests_nonparam.py | ✅ |
| T17.3 | 17 — Inférence V2 | ANCOVA, MANOVA, post-hoc | app/statistics/tests_anova.py | ✅ |
| T17.4 | 17 — Inférence V2 | Tailles d'effet et intervalles de confiance | app/statistics/effect_sizes.py | ✅ |
| T17.5 | 17 — Inférence V2 | Puissance et taille d'échantillon | app/statistics/power.py | ✅ |
| T17.6 | 17 — Inférence V2 | Tests multiples et équivalence (TOST) | app/statistics/multiple_testing.py | ✅ |
| T17.7 | 17 — Inférence V2 | Bootstrap, permutation, puissance par simulation | app/statistics/resampling.py | ✅ |
| T17.8 | 17 — Inférence V2 | Diagnostics d'inférence | app/diagnostics/checks_inference.py | ✅ |
| T17.9 | 17 — Inférence V2 | Tests + validation contre R | tests/test_statistics/ | ✅ |
| T18.1 | 18 — Régression V2 | GLM : probit, Poisson, binomiale négative, Gamma | app/statistics/regression_glm.py | ✅ |
| T18.2 | 18 — Régression V2 | Logistique ordinale | app/statistics/regression_glm.py | ✅ |
| T18.3 | 18 — Régression V2 | Ridge, lasso, elastic net | app/statistics/regression_regularized.py | ✅ |
| T18.4 | 18 — Régression V2 | Régression robuste, polynomiale, splines | app/statistics/regression_robust.py | ⬜ |
| T18.5 | 18 — Régression V2 | Diagnostics de régression V2 | app/diagnostics/checks_regression.py | ⬜ |
| T18.6 | 18 — Régression V2 | Tests + validation contre R | tests/test_statistics/ | ⬜ |
| T19.1 | 19 — Mixtes et longitudinal | Modèles linéaires mixtes | app/statistics/mixed.py | ⬜ |
| T19.2 | 19 — Mixtes et longitudinal | GLMM via R (glmer) | app/r_bridge/scripts/glmm.R | ⬜ |
| T19.3 | 19 — Mixtes et longitudinal | GEE | app/statistics/mixed.py | ⬜ |
| T19.4 | 19 — Mixtes et longitudinal | Diagnostics mixtes (ICC, singularité, convergence) | app/diagnostics/checks_mixed.py | ⬜ |
| T19.5 | 19 — Mixtes et longitudinal | Tests + validation contre R — **Jalon V2-alpha** | tests/test_statistics/ | ⬜ |
| T20.1 | 20 — Multivarié | ACP | app/multivariate/pca.py | ⬜ |
| T20.2 | 20 — Multivarié | Analyse factorielle exploratoire (KMO, Bartlett) | app/multivariate/efa.py | ⬜ |
| T20.3 | 20 — Multivarié | AFC et ACM | app/multivariate/correspondence.py | ⬜ |
| T20.4 | 20 — Multivarié | Clustering (hiérarchique, k-means, DBSCAN, mélanges gaussiens) | app/multivariate/clustering.py | ⬜ |
| T20.5 | 20 — Multivarié | LDA / QDA | app/multivariate/discriminant.py | ⬜ |
| T20.6 | 20 — Multivarié | MDS, corrélation canonique, PLS, ICA, Procruste | app/multivariate/other.py | ⬜ |
| T20.7 | 20 — Multivarié | t-SNE / UMAP (visualisation uniquement) | app/multivariate/embedding.py | ⬜ |
| T20.8 | 20 — Multivarié | CFA / SEM via R (lavaan) | app/r_bridge/scripts/sem.R | ⬜ |
| T20.9 | 20 — Multivarié | Diagnostics + tests + validation | tests/test_multivariate/ | ⬜ |
| T21.1 | 21 — Survie et fiabilité | Kaplan-Meier et log-rank | app/survival/km.py | ⬜ |
| T21.2 | 21 — Survie et fiabilité | Modèle de Cox + diagnostic des risques proportionnels | app/survival/cox.py | ⬜ |
| T21.3 | 21 — Survie et fiabilité | Modèles paramétriques AFT | app/survival/aft.py | ⬜ |
| T21.4 | 21 — Survie et fiabilité | Risques concurrents | app/survival/competing.py | ⬜ |
| T21.5 | 21 — Survie et fiabilité | Fiabilité (Weibull, MTTF, B10) | app/survival/reliability.py | ⬜ |
| T21.6 | 21 — Survie et fiabilité | Frailty via R | app/r_bridge/scripts/frailty.R | ⬜ |
| T21.7 | 21 — Survie et fiabilité | Tests + validation contre R | tests/test_survival/ | ⬜ |
| T22.1 | 22 — Séries temporelles, panel | Décomposition et lissage exponentiel | app/timeseries/decomposition.py | ⬜ |
| T22.2 | 22 — Séries temporelles, panel | Stationnarité, cointégration, causalité de Granger | app/timeseries/stationarity.py | ⬜ |
| T22.3 | 22 — Séries temporelles, panel | ARIMA / SARIMA / SARIMAX | app/timeseries/arima.py | ⬜ |
| T22.4 | 22 — Séries temporelles, panel | VAR / VECM | app/timeseries/var.py | ⬜ |
| T22.5 | 22 — Séries temporelles, panel | GARCH / ARCH | app/timeseries/garch.py | ⬜ |
| T22.6 | 22 — Séries temporelles, panel | Espace d'état / filtre de Kalman | app/timeseries/statespace.py | ⬜ |
| T22.7 | 22 — Séries temporelles, panel | Prévision et backtesting | app/timeseries/forecast.py | ⬜ |
| T22.8 | 22 — Séries temporelles, panel | Panel (effets fixes, aléatoires, Hausman) | app/econometrics/panel.py | ⬜ |
| T22.9 | 22 — Séries temporelles, panel | Variables instrumentales (2SLS, LIML, GMM) | app/econometrics/iv.py | ⬜ |
| T22.10 | 22 — Séries temporelles, panel | Tobit, Heckman, GMM dynamique via R (optionnel) | app/r_bridge/scripts/econometrics.R | ⬜ |
| T22.11 | 22 — Séries temporelles, panel | Tests + validation — **Jalon V2-beta (partie 1)** | tests/test_timeseries/ | ⬜ |
| T23.1 | 23 — Enquêtes et méta-analyse | SurveyDesign + pont R avec données volumineuses | app/survey/design.py, app/r_bridge/bridge.py | ⬜ |
| T23.2 | 23 — Enquêtes et méta-analyse | Estimations pondérées (moyennes, totaux, proportions, domaines) | app/survey/estimates.py | ⬜ |
| T23.3 | 23 — Enquêtes et méta-analyse | Modèles pour enquêtes (svyglm) | app/survey/models.py | ⬜ |
| T23.4 | 23 — Enquêtes et méta-analyse | Calibration, post-stratification, raking | app/survey/calibration.py | ⬜ |
| T23.5 | 23 — Enquêtes et méta-analyse | Estimation de variance (linéarisation, bootstrap, jackknife) | app/survey/variance.py | ⬜ |
| T23.6 | 23 — Enquêtes et méta-analyse | Méta-analyse (effet fixe/aléatoire, I², tau², forest, funnel) | app/meta/meta_analysis.py | ⬜ |
| T23.7 | 23 — Enquêtes et méta-analyse | Biais de publication et méta-régression | app/meta/bias.py | ⬜ |
| T23.8 | 23 — Enquêtes et méta-analyse | Tests + validation contre R — **Jalon V2-beta** | tests/test_survey/, tests/test_meta/ | ⬜ |
| T24.1 | 24 — Causalité | DAG et ensemble d'ajustement (optionnel) | app/causal/dag.py | ⬜ |
| T24.2 | 24 — Causalité | Score de propension + équilibre | app/causal/propensity.py | ⬜ |
| T24.3 | 24 — Causalité | Appariement et IPTW | app/causal/matching.py | ⬜ |
| T24.4 | 24 — Causalité | g-computation | app/causal/gcomp.py | ⬜ |
| T24.5 | 24 — Causalité | Différences de différences | app/causal/did.py | ⬜ |
| T24.6 | 24 — Causalité | Régression sur discontinuité | app/causal/rdd.py | ⬜ |
| T24.7 | 24 — Causalité | Médiation et analyse de sensibilité | app/causal/mediation.py | ⬜ |
| T24.8 | 24 — Causalité | Tests + validation contre R | tests/test_causal/ | ⬜ |
| T25.1 | 25 — ML et interprétabilité | Pipeline ML générique (split, prétraitement, validation croisée) | app/ml/pipeline.py | ⬜ |
| T25.2 | 25 — ML et interprétabilité | Modèles supervisés | app/ml/models.py | ⬜ |
| T25.3 | 25 — ML et interprétabilité | Métriques, ROC/PR, calibration | app/ml/metrics.py | ⬜ |
| T25.4 | 25 — ML et interprétabilité | Recherche d'hyperparamètres (budgétée) | app/ml/tuning.py | ⬜ |
| T25.5 | 25 — ML et interprétabilité | Interprétabilité (permutation, PDP, SHAP) | app/ml/interpret.py | ⬜ |
| T25.6 | 25 — ML et interprétabilité | Déséquilibre de classes et seuil | app/ml/imbalance.py | ⬜ |
| T25.7 | 25 — ML et interprétabilité | Ensembles (stacking, bagging) | app/ml/ensembles.py | ⬜ |
| T25.8 | 25 — ML et interprétabilité | Diagnostics ML + tests | tests/test_ml/ | ⬜ |
| T26.1 | 26 — Reporting étendu | PublicationTable et tableaux de publication | app/reports/publication.py | ⬜ |
| T26.2 | 26 — Reporting étendu | Export Word et LaTeX | app/reports/exporters.py | ⬜ |
| T26.3 | 26 — Reporting étendu | Export PDF | app/reports/exporters.py | ⬜ |
| T26.4 | 26 — Reporting étendu | Export notebook Jupyter / script Quarto | app/reports/exporters.py | ⬜ |
| T26.5 | 26 — Reporting étendu | Rapport de projet multi-analyses | app/reports/generator.py | ⬜ |
| T26.6 | 26 — Reporting étendu | Tests | tests/test_reports/ | ⬜ |
| T27.1 | 27 — Planner V2, agent, UI V2 | Catalogue compact de méthodes | app/agent/catalog.py | ⬜ |
| T27.2 | 27 — Planner V2, agent, UI V2 | Génération des TOOL_SCHEMAS depuis le registre | app/agent/tools.py | ⬜ |
| T27.3 | 27 — Planner V2, agent, UI V2 | Planner V2 (choix en deux étapes) + jeu de 40 questions | app/planner/planner.py | ⬜ |
| T27.4 | 27 — Planner V2, agent, UI V2 | Corrections proposées pour les nouveaux diagnostics | app/agent/orchestrator.py | ⬜ |
| T27.5 | 27 — Planner V2, agent, UI V2 | UI : éditeur de données + journal des transformations + annuler/rétablir | app/ui/ | ⬜ |
| T27.6 | 27 — Planner V2, agent, UI V2 | UI : navigateur de méthodes, projets, sauvegarde automatique | app/ui/ | ⬜ |
| T27.7 | 27 — Planner V2, agent, UI V2 | UI : aide contextuelle | app/ui/ | ⬜ |
| T27.8 | 27 — Planner V2, agent, UI V2 | Tests + test utilisateur structuré — **Jalon V2-RC** | tests/test_agent/, tests/test_planner/ | ⬜ |
| T28.1 | 28 — Optionnel | Régression bayésienne simple + diagnostics de convergence (optionnel) | app/bayes/regression.py | ⬜ |
| T28.2 | 28 — Optionnel | Cartes de contrôle (optionnel) | app/quality/control_charts.py | ⬜ |
| T28.3 | 28 — Optionnel | Plans factoriels et surfaces de réponse (optionnel) | app/quality/doe.py | ⬜ |
| T28.4 | 28 — Optionnel | Tests (optionnel) | tests/test_quality/, tests/test_bayes/ | ⬜ |
| T29.1 | 29 — Stabilisation | Benchmarks de performance (100 k et 1 M lignes) | tests/benchmarks/performance/ | ⬜ |
| T29.2 | 29 — Stabilisation | Audit des faux positifs des diagnostics V2 | tests/test_diagnostics/ | ⬜ |
| T29.3 | 29 — Stabilisation | Documentation : catalogue généré depuis le registre, README, CHANGELOG | docs/, README.md | ⬜ |
| T29.4 | 29 — Stabilisation | Tests utilisateurs V2 (pas de code) | — | ⬜ |
| T29.5 | 29 — Stabilisation | Tag v2.0.0 (pas de code) | CHANGELOG.md | ⬜ |

---

### Phase 15 — Socle V2 + Data Engine V2

**Objectif technique :** poser le registre de méthodes et le journal de transformations, puis étendre le Data Engine (import/export, transformations, manquants, manipulation, gros volumes).
**Dépendances :** `uv add openpyxl pyarrow lxml sqlalchemy` (openpyxl seulement s'il est absent).
**Fichiers à créer :** voir V2 §5 (`app/errors.py`, `app/statistics/registry.py`, `app/data/*`).

**Tâches :**

- **T15.1 — ADR (humain).** Écris toi-même : ADR-003 « Registre de méthodes plutôt qu'extension du Protocol » ; ADR-004 « Dépendances optionnelles par extras » ; ADR-005 « R pour enquêtes, méta-analyse, SEM et frailty ». Gabarit : `ADR-000-template.md`.
- **T15.2 — Socle.** `app/errors.py` (V2 §7.4) ; `app/statistics/registry.py` (V2 §7.1) ; enregistrer dans le registre les méthodes V1 (χ², t, ANOVA, OLS, logistique, MNLogit, Firth, comparaison de modèles) **sans modifier leurs signatures** ; refactorer `generate_reproducible_code` en dispatch par dictionnaire (`register_codegen`), signature publique inchangée.
- **T15.3 — Importeurs étendus** (`app/data/importers.py`, ajouts) :
  ```python
  def import_json(path: str) -> pd.DataFrame: ...
  def import_parquet(path: str) -> pd.DataFrame: ...
  def import_stata(path: str) -> tuple[pd.DataFrame, dict]: ...   # (df, labels) via pandas/pyreadstat
  def import_sas(path: str) -> tuple[pd.DataFrame, dict]: ...
  def import_xml(path: str, xpath: str = "./*") -> pd.DataFrame: ...
  def import_sql(connection_url: str, query: str) -> pd.DataFrame: ...   # SQLAlchemy, SQLite par défaut
  ```
  `import_sql` n'accepte que des requêtes `SELECT` (lever `InvalidMethodInputError` sinon) et n'est jamais appelé avec une requête fournie par un LLM (T15 hérité).
- **T15.4 — Exporteurs** (`app/data/exporters.py`) : `export_csv`, `export_xlsx`, `export_parquet`, `export_stata`, `export_latex_table`, `export_html_table`, tous de forme `(df, path) -> str` (retournent le chemin écrit).
- **T15.5 — Journal de transformations** (`app/data/transform_log.py`) : `TransformStep` et `TransformLog` exacts de V2 §7.2 ; `to_python_script()` produit un script autonome rejouant les étapes.
- **T15.6 — Transformations** (`app/data/transforms.py`) : chaque fonction retourne `(df_nouveau, TransformStep)` et **ne modifie jamais le DataFrame d'entrée** :
  ```python
  def recode_values(df, column: str, mapping: dict, new_column: str | None = None) -> tuple[pd.DataFrame, TransformStep]: ...
  def compute_variable(df, new_column: str, expression: str) -> tuple[pd.DataFrame, TransformStep]: ...
  def standardize(df, columns: list[str]) -> tuple[pd.DataFrame, TransformStep]: ...
  def center(df, columns: list[str]) -> tuple[pd.DataFrame, TransformStep]: ...
  def discretize(df, column: str, bins: int | list[float], labels: list[str] | None = None, strategy: str = "width") -> tuple[pd.DataFrame, TransformStep]: ...
  def extract_date_parts(df, column: str, parts: list[str]) -> tuple[pd.DataFrame, TransformStep]: ...
  ```
  `compute_variable` utilise `DataFrame.eval` avec une liste blanche d'opérateurs ; **jamais** `eval()`/`exec()` Python (T15) ni du texte issu d'un LLM sans validation.
- **T15.7 — Valeurs manquantes** (`app/data/missing.py`) :
  ```python
  def detect_missing(df: pd.DataFrame) -> dict: ...           # taux par colonne, motifs fréquents
  def recode_missing(df, column: str, codes: list) -> tuple[pd.DataFrame, TransformStep]: ...
  def impute_simple(df, column: str, strategy: str) -> tuple[pd.DataFrame, TransformStep]: ...   # mean|median|mode|constant
  def impute_multiple(df, columns: list[str], n_imputations: int = 5, random_state: int = 0) -> list[pd.DataFrame]: ...  # statsmodels.imputation.mice
  def pool_rubin(estimates: list[float], variances: list[float]) -> dict: ...                    # règles de Rubin
  ```
  Produit `HIGH_MISSING_RATE` (seuil 20 % → WARNING, 50 % → PROBABLE_ISSUE, documenter la source). FIML et EM ne sont **pas** implémentés ici : FIML est disponible via lavaan (T20.8).
- **T15.8 — Manipulation** (`app/data/manipulation.py`) : `merge_datasets` (inner/left/right/outer, détection de clés dupliquées), `aggregate_by`, `pivot_longer`, `pivot_wider`, `transpose_dataset`, `drop_duplicates`, `sample_rows` (graine obligatoire). Chaque fonction retourne `(df, TransformStep)`.
- **T15.9 — Gros volumes** (`app/data/large.py`) :
  ```python
  def import_csv_chunked(path: str, chunksize: int = 200_000, usecols: list[str] | None = None) -> pd.DataFrame: ...
  def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame: ...      # int/float/category
  def memory_report(df: pd.DataFrame) -> dict: ...
  def sample_for_analysis(df: pd.DataFrame, max_rows: int, random_state: int = 0) -> tuple[pd.DataFrame, dict]: ...
  ```
  Pas de Dask/Polars/DuckDB (T3, V2 §1.3).
- **T15.10 — Dictionnaire enrichi** (`app/data/dictionary_tools.py`) : `set_variable_metadata(dd, name, *, unit=None, measure_level=None, role=None, weight=None)` met à jour `Variable.metadata` / `Variable.role` / `Variable.weight` (aucun champ nouveau, T17). Niveaux de mesure : `"nominal" | "ordinal" | "interval" | "ratio"`.
- **T15.11 — Tests + fixtures V2.**

**Tests à écrire :** un fichier de test par module ; fixtures : un fichier de chaque format, un dataset avec 30 % de manquants, deux tables à fusionner avec clés dupliquées, un CSV de 500 000 lignes généré par script (marqué `slow`). Tester que les fonctions ne mutent pas leurs entrées.
**Validation de fin de phase :** `TransformLog.to_python_script()` rejoue un enchaînement de 6 transformations et donne un DataFrame identique ; les imports et exports font un aller-retour sans perte sur les 6 formats ; l'import chunké d'un CSV de 1 M de lignes reste sous 4 Go de RAM (T22).

---

### Phase 16 — Descriptif et EDA

**Objectif technique :** couvrir les statistiques descriptives étendues, les corrélations, les tests de normalité, la détection d'outliers et les graphiques exploratoires.
**Dépendances :** aucune nouvelle.

**Tâches :**

- **T16.1 — Fréquences et résumés** (`app/statistics/descriptive.py`, ajouts ; `create_contingency_table` V1 inchangée) :
  ```python
  def frequency_table(df, column: str, weights: str | None = None) -> AnalysisResult: ...
  def summarize_variables(df, columns: list[str] | None = None) -> AnalysisResult: ...
  ```
  `summarize_variables` : n, manquants, moyenne, médiane, mode, écart-type, variance, min, max, étendue, quartiles, IQR, asymétrie, kurtosis. Résultat dans `model_metrics["tables"]["summary"]`.
- **T16.2 — Corrélations** (`app/statistics/correlation.py`) :
  ```python
  def run_correlation(df, variables: list[str], method: str = "pearson") -> AnalysisResult: ...  # pearson|spearman|kendall
  def run_covariance(df, variables: list[str]) -> AnalysisResult: ...
  def run_partial_correlation(df, x: str, y: str, controls: list[str]) -> AnalysisResult: ...
  ```
  Matrices + p-values + IC (transformation de Fisher pour Pearson).
- **T16.3 — Normalité et outliers** (`app/statistics/normality_outliers.py`) :
  ```python
  def run_normality_tests(df, column: str) -> AnalysisResult: ...    # Shapiro-Wilk, D'Agostino, Anderson-Darling
  def detect_outliers(df, columns: list[str], method: str = "iqr") -> AnalysisResult: ...  # iqr|zscore|mahalanobis
  ```
  Émet `NON_NORMAL_DISTRIBUTION` (WARNING) et `OUTLIERS_DETECTED` (INFORMATION). Pour n > 5 000, avertir que Shapiro-Wilk devient hypersensible.
- **T16.4 — Graphiques exploratoires** (`app/reports/plots_eda.py`) — fonctions retournant une `matplotlib.figure.Figure` : `plot_histogram`, `plot_boxplot`, `plot_qq`, `plot_scatter`, `plot_correlation_heatmap`, `plot_pairs` (échantillonner au-delà de 5 000 lignes). Variante interactive Plotly pour `plot_scatter` et `plot_correlation_heatmap`. `generate_plot` V1 n'est pas modifiée.
- **T16.5 — Rapport EDA** (`app/reports/generator.py`, ajout) : `generate_eda_report(df, data_dictionary, output_format="html") -> str` : profil, manquants, résumés, corrélations, outliers, graphiques (images en base64 dans le HTML).
- **T16.6 — Tests + validation contre R** (`summary`, `cor`, `cor.test`, `shapiro.test`).

**Validation de fin de phase :** un rapport EDA complet est généré pour 3 datasets réels dont un « sale » ; les corrélations et tests de normalité concordent avec R à 1e-6.

---

### Phase 17 — Inférence V2

**Objectif technique :** compléter les tests d'hypothèses, tailles d'effet, puissance, corrections multiples et méthodes de rééchantillonnage.
**Dépendances :** `uv add pingouin` (utilisé pour ANCOVA, Welch-ANOVA, Games-Howell et tailles d'effet ; ailleurs, préférer scipy/statsmodels).

**Tâches :**

- **T17.1 — Tests paramétriques** (`app/statistics/tests_parametric.py`) :
  ```python
  def run_paired_t_test(df, var_before: str, var_after: str) -> AnalysisResult: ...
  def run_welch_t_test(df, outcome: str, group: str) -> AnalysisResult: ...
  def run_one_sample_t_test(df, column: str, mu0: float) -> AnalysisResult: ...
  def run_proportion_test(df, column: str, success_value, p0: float = 0.5, group: str | None = None) -> AnalysisResult: ...  # z ou exact
  def run_fisher_exact(df, var1: str, var2: str) -> AnalysisResult: ...
  def run_variance_test(df, outcome: str, group: str, method: str = "levene") -> AnalysisResult: ...  # levene|bartlett|fligner
  ```
- **T17.2 — Non paramétriques** (`app/statistics/tests_nonparam.py`) : `run_mann_whitney`, `run_wilcoxon_signed_rank`, `run_kruskal_wallis`, `run_friedman`, `run_spearman_test`, `run_kendall_test` (signatures `(df, outcome, group)` ou `(df, vars)` selon le test ; exemple complet en V2 §8).
- **T17.3 — ANOVA étendue** (`app/statistics/tests_anova.py`) : `run_ancova(df, outcome, group, covariates)`, `run_manova(df, outcomes, group)` (statsmodels), `run_posthoc(df, outcome, group, method="tukey")` (Tukey HSD, Games-Howell, Dunn avec correction), `run_welch_anova`, `run_repeated_measures_anova` (via pingouin, ou mixtes en Phase 19 si les données sont déséquilibrées).
- **T17.4 — Tailles d'effet** (`app/statistics/effect_sizes.py`) : `cohens_d`, `hedges_g`, `eta_squared`, `partial_eta_squared`, `cramers_v`, `odds_ratio`, `risk_ratio`, `rank_biserial`, chacune avec IC (bootstrap ou formule analytique, documenter). Utilisées par les tests des tâches précédentes pour remplir `model_metrics["effect_size"]`.
- **T17.5 — Puissance** (`app/statistics/power.py`) : `compute_power`, `compute_sample_size`, `compute_detectable_effect` pour t, proportions, ANOVA, corrélation (`statsmodels.stats.power`). Retour : `AnalysisResult` avec `estimates` (puissance, n, effet) et `model_metrics["assumptions"]`.
- **T17.6 — Tests multiples et équivalence** (`app/statistics/multiple_testing.py`) : `adjust_p_values(p_values: dict[str, float], method: str = "holm") -> AnalysisResult` (Bonferroni, Holm, Benjamini-Hochberg, Benjamini-Yekutieli) ; `run_tost_equivalence(df, outcome, group, margin)` ; `run_non_inferiority(df, outcome, group, margin, alternative)`.
- **T17.7 — Rééchantillonnage** (`app/statistics/resampling.py`) :
  ```python
  def run_bootstrap(df, statistic: str, columns: list[str], n_resamples: int = 5000, confidence: float = 0.95, random_state: int = 0) -> AnalysisResult: ...
  def run_permutation_test(df, outcome: str, group: str, statistic: str = "mean_diff", n_resamples: int = 10000, random_state: int = 0) -> AnalysisResult: ...
  def simulate_power(design: dict, n_simulations: int = 2000, random_state: int = 0) -> AnalysisResult: ...
  ```
  `statistic` est une **clé d'une liste blanche** (pas de fonction passée en paramètre, pas d'`eval`). Graine obligatoire.
- **T17.8 — Diagnostics d'inférence** (`app/diagnostics/checks_inference.py`) : `check_equal_variances`, `check_expected_counts`, `check_group_sizes`, appelés par les tests concernés. Codes : `UNEQUAL_VARIANCES`, `LOW_EXPECTED_COUNTS`, `SMALL_GROUP_SIZE` (V2 §7.6).
- **T17.9 — Tests + validation contre R** (`t.test`, `wilcox.test`, `kruskal.test`, `friedman.test`, `fisher.test`, `prop.test`, `p.adjust`, `power.t.test`, `TukeyHSD`, `car::Anova` pour ANCOVA).

**Tolérances :** `rel=1e-6` pour les tests de forme fermée ; bootstrap et permutations : graine fixe + vérifier que la valeur de R tombe dans l'IC et que l'ordre de grandeur de la p-value concorde.
**Validation de fin de phase :** les 9 tâches respectent la Définition de Done (V2 §9.1) ; le corpus de cas limites contient au moins un cas déclenchant chacun des trois diagnostics et aucun faux positif sur le dataset propre.

---

### Phase 18 — Régression V2

**Objectif technique :** GLM supplémentaires, régressions régularisée/robuste/polynomiale, diagnostics de régression étendus.
**Dépendances :** `uv add scikit-learn`.

**Tâches :**

- **T18.1 — GLM** (`app/statistics/regression_glm.py`) :
  ```python
  def run_probit_regression(df, outcome: str, predictors: list[str]) -> AnalysisResult: ...
  def run_poisson_regression(df, outcome: str, predictors: list[str], exposure: str | None = None) -> AnalysisResult: ...
  def run_negative_binomial_regression(df, outcome: str, predictors: list[str], exposure: str | None = None) -> AnalysisResult: ...
  def run_gamma_regression(df, outcome: str, predictors: list[str], link: str = "log") -> AnalysisResult: ...
  ```
  `statsmodels.api.GLM`. Rapports de taux d'incidence (`exp(coef)`) dans `model_metrics` pour Poisson/NB.
- **T18.2 — Logistique ordinale** : `run_ordinal_regression(df, outcome, predictors, distr="logit")` via `statsmodels.miscmodels.ordinal_model.OrderedModel` (vérifier l'API installée). Vérifier que l'ordre des modalités est fourni par `Variable.levels`, sinon lever `InvalidMethodInputError`.
- **T18.3 — Régularisée** (`app/statistics/regression_regularized.py`) : `run_ridge`, `run_lasso`, `run_elastic_net` avec choix de la pénalité par validation croisée (scikit-learn `*CV`, graine fixe). `p_values = {}` + avertissement (V2 §2.2). Standardisation interne des prédicteurs, coefficients restitués dans l'échelle d'origine ET standardisée. Consigner la pénalité retenue dans `execution_log`.
- **T18.4 — Robuste, polynomiale, splines** (`app/statistics/regression_robust.py`) : `run_robust_regression` (statsmodels `RLM`, norme de Huber), `run_polynomial_regression(df, outcome, predictor, degree)` et `run_spline_regression(df, outcome, predictor, df_spline)` (formules patsy `bs()`), avec courbe ajustée dans `model_metrics["tables"]["fitted_curve"]`.
- **T18.5 — Diagnostics** (`app/diagnostics/checks_regression.py`) :
  ```python
  def check_heteroscedasticity(model_result) -> list[DiagnosticFinding]: ...   # Breusch-Pagan, White
  def check_residual_normality(model_result) -> list[DiagnosticFinding]: ...    # Shapiro / Jarque-Bera
  def check_autocorrelation(model_result) -> list[DiagnosticFinding]: ...       # Durbin-Watson
  def check_overdispersion(model_result) -> list[DiagnosticFinding]: ...        # Poisson : Pearson χ²/ddl
  def check_convergence(model_result) -> list[DiagnosticFinding]: ...
  ```
  Brancher ces diagnostics sur OLS, logistique (V1) et les nouveaux GLM **sans changer les signatures V1** (ajout de diagnostics dans `result.diagnostics`).
- **T18.6 — Tests + validation contre R** (`glm(family=poisson/Gamma/binomial(link="probit"))`, `MASS::glm.nb`, `MASS::polr`, `glmnet` pour lasso/ridge à pénalité fixée, `MASS::rlm`, `lmtest::bptest`).

**Tolérances :** GLM `rel=1e-4` ; régularisée : comparer à `glmnet` avec **λ imposé** (les chemins de pénalité diffèrent d'un outil à l'autre) ; robuste `rel=1e-4`.
**Validation de fin de phase :** chaque diagnostic se déclenche sur son cas dédié (hétéroscédasticité construite, surdispersion construite) et pas sur le dataset propre.

---

### Phase 19 — Modèles mixtes et longitudinal

**Objectif technique :** modèles à effets aléatoires, données longitudinales et GEE.
**Dépendances :** aucune nouvelle côté Python ; R : `lme4`, `geepack` (références de validation et GLMM).

**Tâches :**

- **T19.1 — Modèles linéaires mixtes** (`app/statistics/mixed.py`) :
  ```python
  def run_linear_mixed_model(df, outcome: str, fixed: list[str], group: str,
                             random_slopes: list[str] | None = None, reml: bool = True) -> AnalysisResult: ...
  ```
  `statsmodels.formula.api.mixedlm`. `model_metrics` : variances des effets aléatoires, ICC, AIC/BIC (si ML).
- **T19.2 — GLMM via R** (`app/r_bridge/scripts/glmm.R`) + `run_glmm(df, outcome, fixed, group, family)` : `lme4::glmer` via le pont (R requis, dégradation propre, T20). Pour des données de plus de 50 000 lignes, utiliser `call_r_script_with_data` (créée en T23.1) ; avant cela, limiter à 50 000 lignes et le signaler.
- **T19.3 — GEE** : `run_gee(df, outcome, predictors, group, family="gaussian", cov_struct="exchangeable")` via `statsmodels.genmod.GEE`.
- **T19.4 — Diagnostics** (`app/diagnostics/checks_mixed.py`) : `check_singular_fit` (variance d'effet aléatoire ≈ 0), `check_mixed_convergence`, `check_group_structure` (nombre de groupes < 5 → avertissement ; groupes de taille 1).
- **T19.5 — Tests + validation contre R** (`lme4::lmer`, `lme4::glmer`, `geepack::geeglm`). **Jalon V2-alpha.**

**Validation de fin de phase :** un modèle à intercept aléatoire, un à pente aléatoire, un GLMM et un GEE sont validés contre R (tolérance `rel=1e-3` sur les effets fixes, justifiée : optimiseurs différents) ; `SINGULAR_FIT` est déclenché sur un cas construit. **Jalon V2-alpha :** tag `v1.1.0`, rapport de validation à jour.

---

### Phase 20 — Analyse multivariée

**Objectif technique :** réduction de dimension, analyses factorielles, clustering, analyse discriminante.
**Dépendances :** `uv add --optional multivariate factor-analyzer prince` ; `uv add --optional embedding umap-learn` (optionnel). Les méthodes de scikit-learn (installé en Phase 18) servent pour ACP, clustering, LDA/QDA, MDS, PLS, ICA.

**Règle propre à cette phase (T24) :** chaque résultat exploratoire ajoute dans `warnings` ce qu'il ne permet pas de conclure (« les clusters sont une description, pas une preuve d'existence de groupes réels », « les axes de t-SNE/UMAP ne sont pas interprétables en distance »).

**Tâches :**

- **T20.1 — ACP** (`app/multivariate/pca.py`) : `run_pca(df, variables, n_components=None, standardize=True)` ; variance expliquée, charges, scores (`model_metrics["tables"]`), éboulis via `plot_scree`. Le choix automatique du nombre de composantes (critère de Kaiser ou 80 % de variance) est consigné.
- **T20.2 — Analyse factorielle exploratoire** (`app/multivariate/efa.py`) : `run_efa(df, variables, n_factors, rotation="varimax")` avec KMO et test de sphéricité de Bartlett (`factor_analyzer`) ; produit `LOW_KMO` si KMO < 0,6.
- **T20.3 — AFC et ACM** (`app/multivariate/correspondence.py`) : `run_correspondence_analysis(df, row, col)` et `run_mca(df, variables)` via `prince`.
- **T20.4 — Clustering** (`app/multivariate/clustering.py`) :
  ```python
  def run_hierarchical_clustering(df, variables, n_clusters: int, linkage: str = "ward") -> AnalysisResult: ...
  def run_kmeans(df, variables, n_clusters: int | None = None, k_max: int = 10, random_state: int = 0) -> AnalysisResult: ...
  def run_dbscan(df, variables, eps: float, min_samples: int) -> AnalysisResult: ...
  def run_gaussian_mixture(df, variables, n_components: int | None = None, random_state: int = 0) -> AnalysisResult: ...
  ```
  Si `n_clusters` est `None`, choisir par silhouette (ou BIC pour les mélanges) et **consigner le choix** (V2 §2.2). Échantillonner au-delà de 20 000 lignes pour DBSCAN et le clustering hiérarchique (T22).
- **T20.5 — Discriminante** (`app/multivariate/discriminant.py`) : `run_lda`, `run_qda` avec validation croisée stratifiée ; matrice de confusion dans `model_metrics["tables"]`.
- **T20.6 — Méthodes secondaires** (`app/multivariate/other.py`) : `run_mds`, `run_canonical_correlation`, `run_pls_regression`, `run_ica`, `run_procrustes` (scipy).
- **T20.7 — t-SNE / UMAP** (`app/multivariate/embedding.py`) : `run_tsne` (scikit-learn), `run_umap` (extra `embedding`). **Visualisation uniquement** : aucun résultat de ces méthodes ne doit alimenter un test d'hypothèse.
- **T20.8 — CFA / SEM via R** (`app/r_bridge/scripts/sem.R`, `app/multivariate/sem.py`) : `run_cfa(df, model_syntax: str, estimator="ML", missing="listwise")` et `run_sem(...)` via `lavaan` ; `missing="fiml"` disponible. La syntaxe lavaan est fournie par l'utilisateur (ou validée par lui après proposition de l'IA, T14), jamais exécutée sans confirmation.
- **T20.9 — Diagnostics + tests + validation.**

**Tests et validation :** références R : `prcomp`, `psych::fa`, `psych::KMO`, `FactoMineR::CA`/`MCA`, `stats::hclust`, `MASS::lda`/`qda`, `lavaan`. Tolérances : ACP et LDA `rel=1e-6` (au signe près des axes, tester `abs()`) ; k-means et mélanges : propriétés (nombre de clusters, ARI ≥ 0,9 avec une partition de référence construite) ; EFA : charges à `1e-3` après alignement des rotations.
**Validation de fin de phase :** les 9 tâches respectent la Définition de Done ; `LOW_KMO` se déclenche sur un cas construit.

---

### Phase 21 — Survie et fiabilité

**Objectif technique :** analyse de durées, censure, modèles de survie et fiabilité.
**Dépendances :** `uv add --optional survival lifelines` ; R : `survival`, `coxme` (références + frailty).

**Convention de données :** `Variable.role = "duration"` pour le temps, `"event"` pour l'indicateur (1 = événement, 0 = censuré) ; si les rôles ne sont pas assignés, lever `InvalidMethodInputError`.

**Tâches :**

- **T21.1 — Kaplan-Meier et log-rank** (`app/survival/km.py`) : `run_kaplan_meier(df, duration, event, group=None)` ; `run_logrank_test(df, duration, event, group)` ; table de survie dans `model_metrics["tables"]["survival_table"]`, médiane de survie avec IC. Graphique `plot_survival_curves`.
- **T21.2 — Cox** (`app/survival/cox.py`) : `run_cox_ph(df, duration, event, predictors, strata=None)` ; hazard ratios + IC. Diagnostic `check_proportional_hazards` (résidus de Schoenfeld) → `PROPORTIONAL_HAZARDS_VIOLATED`. Gérer la troncature à gauche via un paramètre `entry: str | None`.
- **T21.3 — AFT paramétriques** (`app/survival/aft.py`) : `run_aft(df, duration, event, predictors, distribution="weibull")` (weibull, lognormal, loglogistic) ; AIC pour comparer.
- **T21.4 — Risques concurrents** (`app/survival/competing.py`) : `run_cumulative_incidence(df, duration, event_type, event_of_interest, group=None)` (Aalen-Johansen via lifelines).
- **T21.5 — Fiabilité** (`app/survival/reliability.py`) : `run_weibull_reliability(df, duration, event)` : paramètres de forme/échelle, MTTF, vie B10, courbe de fiabilité.
- **T21.6 — Frailty via R** (`app/r_bridge/scripts/frailty.R`, `run_cox_frailty(df, duration, event, predictors, cluster)`) : `survival::coxph` avec terme `frailty()` ou `coxme`.
- **T21.7 — Tests + validation contre R** (`survival::survfit`, `survdiff`, `coxph`, `survreg`, `cmprsk::cuminc`). Tolérances : KM et log-rank `rel=1e-6` ; Cox/AFT `rel=1e-4`.

**Validation de fin de phase :** KM, log-rank, Cox, AFT validés contre R ; `PROPORTIONAL_HAZARDS_VIOLATED` se déclenche sur un cas construit (effet qui varie dans le temps) et pas sur un dataset propre.

---

### Phase 22 — Séries temporelles, panel et économétrie

**Objectif technique :** modélisation et prévision de séries, données de panel, variables instrumentales.
**Dépendances :** `uv add --optional timeseries linearmodels arch` ; R : `forecast`, `urca`, `tseries`, `plm`, `AER`, `sampleSelection` (références et tâche T22.10).

**Convention de données :** une série est une colonne numérique + une colonne temporelle ordonnée (`dtype = "datetime"`) avec fréquence déclarée (`"D"`, `"W"`, `"M"`, `"Q"`, `"Y"`) ; si la fréquence est absente ou irrégulière, lever `InvalidMethodInputError`.

**Tâches :**

- **T22.1 — Décomposition et lissage** (`app/timeseries/decomposition.py`) : `run_stl_decomposition(df, time, value, period)`, `run_holt_winters(df, time, value, seasonal_periods, trend, seasonal)` (statsmodels).
- **T22.2 — Stationnarité et causalité** (`app/timeseries/stationarity.py`) : `run_adf_test`, `run_kpss_test`, `run_cointegration_test(df, time, variables)` (Engle-Granger/Johansen), `run_granger_causality(df, time, cause, effect, max_lag)`. Produit `NON_STATIONARY_SERIES`. **Avertir** que Granger ne prouve pas une causalité au sens causal (T24).
- **T22.3 — ARIMA** (`app/timeseries/arima.py`) : `run_arima(df, time, value, order, seasonal_order=None, exog=None)` ; si `order=None`, recherche par AIC sur une grille **bornée** (p,q ≤ 3, d ≤ 2) consignée dans `execution_log`. Diagnostics des résidus (Ljung-Box → `AUTOCORRELATED_RESIDUALS`).
- **T22.4 — VAR / VECM** (`app/timeseries/var.py`) : `run_var(df, time, variables, max_lag)`, `run_vecm(df, time, variables, coint_rank, k_ar_diff)`, fonctions de réponse impulsionnelle dans `model_metrics["tables"]`.
- **T22.5 — GARCH** (`app/timeseries/garch.py`) : `run_garch(df, time, value, p=1, q=1, mean="Constant", dist="normal")` via `arch`.
- **T22.6 — Espace d'état / Kalman** (`app/timeseries/statespace.py`) : `run_local_level_model`, `run_unobserved_components(df, time, value, level, trend, seasonal)` via `statsmodels.tsa.statespace`.
- **T22.7 — Prévision et backtesting** (`app/timeseries/forecast.py`) : `forecast_series(result_ref, horizon)` et `run_backtest(df, time, value, model: str, initial_window, horizon, step)` (origine glissante, **jamais** de validation croisée aléatoire sur une série temporelle). Métriques : MAE, RMSE, MAPE (avec garde-fou si zéros), MASE.
- **T22.8 — Panel** (`app/econometrics/panel.py`) : `run_panel_fixed_effects`, `run_panel_random_effects`, `run_first_difference`, `run_hausman_test(df, entity, time, outcome, predictors)` via `linearmodels`. Erreurs standard robustes/groupées par entité en option. Le test de Hausman, absent de `linearmodels`, est calculé à partir des deux estimations (formule documentée, c'est l'**exception documentée** à T7) et validé contre `plm::phtest`.
- **T22.9 — Variables instrumentales** (`app/econometrics/iv.py`) : `run_iv_2sls`, `run_iv_liml`, `run_iv_gmm(df, outcome, endogenous, instruments, exogenous)` via `linearmodels.iv` ; tests de première étape, Sargan/Hansen, Wu-Hausman ; `WEAK_INSTRUMENT` si F de première étape < 10 (source : Stock & Yogo, documenter).
- **T22.10 — Tobit, Heckman, GMM dynamique via R (optionnel)** : `run_tobit` (`AER::tobit`), `run_heckman` (`sampleSelection::heckit`), `run_dynamic_panel_gmm` (`plm::pgmm`) via le pont. Marquer ces méthodes `engine="r"` dans le registre.
- **T22.11 — Tests + validation contre R** (`forecast::Arima`, `stats::HoltWinters`, `tseries::adf.test`, `urca::ca.jo`, `plm::plm`, `AER::ivreg`). Tolérances : ARIMA/GARCH/VAR `rel=1e-3` sur les paramètres (optimiseurs différents, à justifier) ; panel et IV `rel=1e-6`.

**Validation de fin de phase :** chaque méthode est validée ; le backtesting n'utilise jamais de données futures (test dédié sur série construite) ; `NON_STATIONARY_SERIES` et `WEAK_INSTRUMENT` se déclenchent sur des cas construits. **Jalon V2-beta (partie 1).**

---

### Phase 23 — Enquêtes complexes et méta-analyse

**Objectif technique :** estimation sur plans de sondage via R (`survey`) et méta-analyse.
**Dépendances :** aucune nouvelle côté Python ; R : `survey`, `metafor`.

**Tâches :**

- **T23.1 — `SurveyDesign` + pont R pour gros volumes** (`app/survey/design.py`, `app/r_bridge/bridge.py`) : `SurveyDesign` exact de V2 §7.3 ; `validate_survey_design(df, design) -> list[DiagnosticFinding]` (poids ≤ 0, strates avec une seule UPE, valeurs manquantes dans les poids). Ajouter dans `bridge.py` une **nouvelle** fonction (sans toucher à `call_r_script`) :
  ```python
  def call_r_script_with_data(script_name: str, df: pd.DataFrame, params: dict, timeout: int = 300) -> dict:
      """Écrit df dans un fichier CSV temporaire (jamais en JSON), passe les chemins
      d'entrée/sortie à Rscript et lit le JSON de sortie. Nettoie les fichiers temporaires."""
  ```
- **T23.2 — Estimations pondérées** (`app/survey/estimates.py`) : `run_survey_mean`, `run_survey_total`, `run_survey_proportion`, `run_survey_crosstab`, chacune acceptant `domain` pour les sous-populations (`svyby`) ; effets de plan (`deff`) dans `model_metrics` ; `HIGH_DESIGN_EFFECT` si `deff` > 2 (documenter la source).
- **T23.3 — Modèles pour enquêtes** (`app/survey/models.py`) : `run_survey_glm(df, design, outcome, predictors, family)` via `svyglm`.
- **T23.4 — Calibration** (`app/survey/calibration.py`) : `run_poststratification`, `run_raking`, `run_calibration` ; poids ajustés retournés dans `model_metrics["tables"]["weights_summary"]` (résumé, pas les poids ligne à ligne) et DataFrame séparé via une fonction `apply_calibrated_weights`.
- **T23.5 — Variance** (`app/survey/variance.py`) : linéarisation (Taylor, défaut), bootstrap et jackknife par poids répliqués (`as.svrepdesign`) ; paramètre `variance_method` explicite.
- **T23.6 — Méta-analyse** (`app/meta/meta_analysis.py`) : `run_meta_analysis(df, effect: str, variance: str | None, se: str | None, study_labels: str, model="random", method="REML")` ; Python natif (`statsmodels.stats.meta_analysis`) pour effet fixe et DerSimonian-Laird, R `metafor::rma` pour REML et autres estimateurs. Résultats : effet combiné, I², τ², Q ; `HIGH_HETEROGENEITY` si I² > 75 % (documenter la source). Graphiques `plot_forest`, `plot_funnel`.
- **T23.7 — Biais de publication et méta-régression** (`app/meta/bias.py`) : `run_egger_test`, `run_trim_and_fill`, `run_meta_regression(df, ..., moderators)` via `metafor`. Avertir que ces tests manquent de puissance avec moins de 10 études.
- **T23.8 — Tests + validation contre R** (`survey::svymean`, `svyglm`, `postStratify`, `rake`, `metafor::rma`, `regtest`, `trimfill`). Tolérances : estimations ponctuelles `rel=1e-6` ; variances `rel=1e-4`.

**Validation de fin de phase :** un jeu d'enquête à plan stratifié en grappes est analysé avec poids et domaines et concorde avec R ; une méta-analyse de 10 études concorde avec `metafor` ; les deux diagnostics se déclenchent sur cas construits. **Jalon V2-beta :** tag `v1.2.0`.

---

### Phase 24 — Inférence causale

**Objectif technique :** méthodes d'estimation d'effets causaux sous hypothèses explicites.
**Dépendances :** `uv add --optional causal rdrobust networkx` ; R : `MatchIt`, `rdrobust` (références).

**Règle propre à cette phase (T24) :** toute méthode causale ajoute dans `warnings` les **hypothèses d'identification** qu'elle exige (pas de confusion non mesurée, chevauchement, tendances parallèles, continuité à la discontinuité…) et ne présente jamais son résultat comme une preuve causale. L'IA n'a pas le droit de décider si ces hypothèses sont plausibles : elle les affiche, l'utilisateur tranche.

**Tâches :**

- **T24.1 — DAG et ensemble d'ajustement (optionnel)** (`app/causal/dag.py`) : `find_backdoor_adjustment_set(edges: list[tuple[str, str]], treatment: str, outcome: str) -> list[str]` avec `networkx`. Le DAG est fourni par l'utilisateur ; la fonction vérifie l'absence de cycle. À livrer avec précaution (cas complexes) — si la complexité dépasse les capacités de la tâche, la reporter plutôt que de la livrer fausse.
- **T24.2 — Score de propension et équilibre** (`app/causal/propensity.py`) : `estimate_propensity_scores(df, treatment, covariates)` (logit statsmodels) ; `check_covariate_balance(df, treatment, covariates, weights=None)` → différences standardisées (SMD) ; `POOR_COVARIATE_BALANCE` si |SMD| > 0,1 (source : Austin 2009, documenter). `check_overlap` signale les scores proches de 0 ou 1.
- **T24.3 — Appariement et IPTW** (`app/causal/matching.py`) : `run_propensity_matching(df, treatment, outcome, covariates, caliper=0.2, ratio=1, random_state=0)` (plus proche voisin, scikit-learn) ; `run_iptw(df, treatment, outcome, covariates, stabilized=True, trim=None)` ; effets ATE/ATT avec IC bootstrap ; équilibre avant/après dans `model_metrics["tables"]`.
- **T24.4 — g-computation** (`app/causal/gcomp.py`) : `run_g_computation(df, treatment, outcome, covariates, family)`.
- **T24.5 — Différences de différences** (`app/causal/did.py`) : `run_did(df, outcome, treated, post, covariates=None, cluster=None)` (OLS avec interaction, erreurs groupées) ; `run_event_study(df, outcome, entity, time, treated_group, event_time)` ; test de pré-tendances.
- **T24.6 — Discontinuité** (`app/causal/rdd.py`) : `run_rdd(df, outcome, running_variable, cutoff, bandwidth=None)` via `rdrobust` ; test de manipulation de la variable d'affectation (densité) en avertissement.
- **T24.7 — Médiation et sensibilité** (`app/causal/mediation.py`) : `run_mediation(df, outcome, treatment, mediator, covariates)` (`statsmodels.stats.mediation.Mediation`) ; `compute_e_value(estimate, ci_bound, kind)` (formule publiée, documenter la référence).
- **T24.8 — Tests + validation** (`MatchIt`, `WeightIt` si disponible, `rdrobust`, simulations à effet vrai connu). **Particularité :** en plus de R, tester sur des **données simulées à effet causal connu** : l'estimateur doit retrouver l'effet vrai dans un intervalle raisonnable.

**Reportés (V3) :** contrôles synthétiques, TMLE, modèles structurels causaux complets.
**Validation de fin de phase :** les méthodes retrouvent l'effet simulé et concordent avec R lorsqu'une référence existe ; `POOR_COVARIATE_BALANCE` se déclenche sur un cas construit.

---

### Phase 25 — Machine learning et interprétabilité

**Objectif technique :** modèles prédictifs classiques avec validation rigoureuse et interprétabilité, sans deep learning.
**Dépendances :** `uv add --optional ml shap imbalanced-learn` (scikit-learn déjà installé en Phase 18).

**Règles propres à cette phase :** (1) toute évaluation se fait sur des données **non vues** (jeu de test ou validation croisée) ; (2) `random_state` obligatoire partout ; (3) `p_values = {}` et un avertissement : « un modèle prédictif ne permet pas de conclure à une relation causale » (T24) ; (4) pas de modèle lourd en mémoire (T3) : réseaux de neurones limités à `MLPClassifier`/`MLPRegressor` de scikit-learn, petite taille.

**Tâches :**

- **T25.1 — Pipeline générique** (`app/ml/pipeline.py`) :
  ```python
  def split_data(df, target: str, test_size: float = 0.2, stratify: bool = True, random_state: int = 0) -> dict: ...
  def build_preprocessor(data_dictionary: "DataDictionary", predictors: list[str]) -> "ColumnTransformer": ...  # imputation, encodage, standardisation, DANS le pipeline pour éviter les fuites
  def cross_validate_model(df, target: str, predictors: list[str], model_id: str, task: str, cv: int = 5, random_state: int = 0) -> AnalysisResult: ...
  ```
  `task` : `"classification"` | `"regression"`. Toutes les transformations apprises (imputation, standardisation) sont **apprises sur le jeu d'entraînement uniquement**.
- **T25.2 — Modèles supervisés** (`app/ml/models.py`) : registre interne `MODEL_CATALOG` (liste blanche) avec : `logistic`, `linear`, `decision_tree`, `random_forest`, `gradient_boosting` (`HistGradientBoosting*` par défaut), `svm`, `knn`, `naive_bayes`, `mlp_small`. `run_ml_model(df, target, predictors, model_id, task, params=None, random_state=0) -> AnalysisResult` ; `model_id` hors catalogue → `InvalidMethodInputError`.
- **T25.3 — Métriques** (`app/ml/metrics.py`) : classification : accuracy, précision, rappel, F1, AUC-ROC, AUC-PR, log-loss, Brier, matrice de confusion, courbe de calibration ; régression : MAE, RMSE, R², erreur relative médiane. Graphiques `plot_roc`, `plot_pr`, `plot_calibration`, `plot_confusion_matrix`.
- **T25.4 — Hyperparamètres** (`app/ml/tuning.py`) : `tune_hyperparameters(df, target, predictors, model_id, task, n_iter=20, cv=5, random_state=0)` avec `RandomizedSearchCV` et des **grilles prédéfinies par modèle** (jamais générées par un LLM) ; budget de temps maximal (`timeout`) pour respecter T3 ; paramètres retenus consignés.
- **T25.5 — Interprétabilité** (`app/ml/interpret.py`) : `compute_permutation_importance`, `compute_partial_dependence(df, ..., feature)`, `compute_shap_values(df, ..., max_rows=1000)` (échantillonnage obligatoire, T22 ; modèles d'arbres uniquement par défaut). Résultats dans `estimates` (importances) et `model_metrics["tables"]`.
- **T25.6 — Déséquilibre** (`app/ml/imbalance.py`) : `detect_class_imbalance` (→ `CLASS_IMBALANCE` si la classe minoritaire < 10 %), `apply_resampling` (SMOTE, sous-échantillonnage, **appliqué uniquement dans les plis d'entraînement**), `tune_decision_threshold` (sur validation, pas sur test).
- **T25.7 — Ensembles** (`app/ml/ensembles.py`) : `run_stacking`, `run_bagging` (scikit-learn).
- **T25.8 — Diagnostics ML + tests** (`app/diagnostics/checks_ml.py`) : `check_leakage` (variable quasi parfaitement corrélée à la cible, identifiant, ou variable post-événement signalée → `LEAKAGE_SUSPECTED`), `check_overfitting` (écart apprentissage/validation > 15 points de métrique → `OVERFITTING_SUSPECTED`, seuil documenté). Tests : référence scikit-learn exécutée indépendamment (R n'est pas la référence ici : documenter, T12 adapté en V2 §9.1 point 5) ; test de **non-fuite** : le prétraitement n'utilise jamais le jeu de test.

**Validation de fin de phase :** un pipeline complet (préparation → CV → métriques → importance) est reproductible à la graine près (deux exécutions donnent des résultats identiques) ; `LEAKAGE_SUSPECTED` se déclenche sur un jeu avec une variable-cible déguisée ; aucune fonction ne dépasse 4 Go sur 1 M de lignes (échantillonnage signalé).

---

### Phase 26 — Reporting étendu

**Objectif technique :** tableaux de publication, exports Word/LaTeX/PDF et notebooks.
**Dépendances :** `uv add --optional reporting python-docx nbformat weasyprint` + `sudo apt install -y libpango-1.0-0 libpangoft2-1.0-0`.

**Tâches :**

- **T26.1 — Tableaux de publication** (`app/reports/publication.py`) :
  ```python
  @dataclass
  class PublicationTable:
      title: str
      headers: list[str]
      rows: list[list[str]]       # cellules déjà formatées (texte)
      notes: list[str] = field(default_factory=list)

  def build_regression_table(results: list[AnalysisResult], style: str = "apa") -> PublicationTable: ...
  def build_descriptive_table(result: AnalysisResult, style: str = "apa") -> PublicationTable: ...
  def build_comparison_table(results: list[AnalysisResult], style: str = "apa") -> PublicationTable: ...
  ```
  Règles de formatage (p < 0,001, décimales, étoiles de signification optionnelles) définies dans le code, jamais par un LLM.
- **T26.2 — Word et LaTeX** (`app/reports/exporters.py`) : `export_table_docx(table, path)`, `export_report_docx(result, path)` (python-docx), `export_table_latex(table) -> str`, `export_report_latex(result) -> str`. Échapper les caractères spéciaux LaTeX ; tester la compilation d'un exemple si `pdflatex` est présent (test `skipif`).
- **T26.3 — PDF** : `export_report_pdf(result, path)` via HTML → WeasyPrint (import paresseux, T18). Dégradation propre si les bibliothèques système manquent (`DependencyMissingError` avec la commande `apt`).
- **T26.4 — Notebook et script reproductible** : `export_notebook(results: list[AnalysisResult], path)` (`nbformat`) : une cellule Markdown par analyse (méthode, hypothèses, diagnostics) + une cellule de code (`generated_code`) ; `export_quarto(results, path)` génère un `.qmd`. Les notebooks sont **générés, jamais exécutés** par l'application (T15).
- **T26.5 — Rapport de projet** (`app/reports/generator.py`, ajout) : `generate_project_report(project_dir: str, analysis_ids: list[str], output_format: str = "html") -> str` à partir des analyses sauvegardées (Phase 11 V1) : sommaire, méthodes, résultats, diagnostics, journal des transformations (`TransformLog`), versions logicielles.
- **T26.6 — Tests.** Vérifier que le `.docx` s'ouvre (relecture par python-docx), que le LaTeX échappe correctement `%`, `_`, `&`, `#`, que le notebook est valide (`nbformat.validate`).

**Validation de fin de phase :** pour 5 méthodes de familles différentes, les 4 formats sont produits et relus sans erreur.

---

### Phase 27 — Planner V2, agent et UI V2

**Objectif technique :** rendre toutes les méthodes accessibles à l'utilisateur et à l'IA, sans saturer le contexte du LLM.
**Dépendances :** aucune nouvelle.

**Principe de conception (important) :** avec une centaine de méthodes, exposer un schéma de tool calling par méthode **au même moment** noierait le LLM. Le planner V2 raisonne donc **en deux étapes** : (1) choisir une *famille* et une *méthode* dans un **catalogue compact filtré** ; (2) seulement ensuite, recevoir le schéma détaillé de **cette seule méthode**. Le LLM ne voit toujours que des métadonnées (V1 §2).

**Tâches :**

- **T27.1 — Catalogue compact** (`app/agent/catalog.py`) :
  ```python
  def get_method_catalog(data_dictionary: "DataDictionary", families: list[str] | None = None,
                         available_extras: set[str] | None = None) -> list[dict]: ...
  ```
  Retourne, pour chaque méthode du registre compatible avec les types de variables et la taille d'échantillon : `method_id`, `label_fr`, `family`, `assumptions` (courtes), `engine`. Exclut les méthodes dont l'extra ou R est indisponible (et pas de message d'erreur : elles sont simplement absentes).
- **T27.2 — `TOOL_SCHEMAS` générés** (`app/agent/tools.py`) : `build_tool_schema(method_id: str) -> dict` génère le schéma JSON depuis la signature + `MethodSpec` ; `execute_tool` route via `get_method` (signature publique inchangée). Le garde-fou V1 (aucun `DataFrame` brut reçu du LLM) est conservé et **re-testé** pour chaque famille.
- **T27.3 — Planner V2** (`app/planner/planner.py`) : `build_analysis_plan` (signature V1 inchangée) s'appuie sur `get_method_catalog` ; étape 1 → choix de famille et méthode ; étape 2 → remplissage des paramètres validé par `AnalysisPlan` (Pydantic). `confidence="low"` plutôt qu'un plan hasardeux ; ne jamais choisir une méthode absente du catalogue filtré. Jeu de **40 questions réelles** (constitué par toi), couvrant toutes les familles.
- **T27.4 — Corrections étendues** (`app/agent/orchestrator.py`) : `propose_correction` gère les nouveaux diagnostics selon une **table de correspondance déterministe** (ex. `OVERDISPERSION` → proposer binomiale négative ; `PROPORTIONAL_HAZARDS_VIOLATED` → proposer Cox stratifié ou AFT ; `UNEQUAL_VARIANCES` → proposer Welch ; `NON_STATIONARY_SERIES` → proposer différenciation). Toujours `requires_confirmation=True` (T14).
- **T27.5 — UI : éditeur et journal** (`app/ui/`) : `st.data_editor` pour modifier des cellules ; les modifications sont enregistrées dans le `TransformLog` ; boutons annuler/rétablir basés sur le journal. Pas de tableur complet (V2 §1.3).
- **T27.6 — UI : méthodes et projets** : navigateur de méthodes (par famille, avec hypothèses et extras requis), gestion de projets (ouvrir/enregistrer via `app/reproducibility`), sauvegarde automatique après chaque analyse.
- **T27.7 — UI : aide contextuelle** : panneau d'aide alimenté par `MethodSpec` (hypothèses, référence R, diagnostics possibles) ; pas de texte généré par un LLM pour cette aide.
- **T27.8 — Tests + test utilisateur structuré** : tests unitaires de `get_method_catalog` (filtrage par types, taille, extras absents), de `build_tool_schema` pour **chaque** méthode du registre (test paramétré), du planner (≥ 75 % de plans corrects sur 40 questions ; le reste déclenche `needs_clarification`) ; test manuel : un testeur externe réalise 3 analyses de familles différentes sans aide. **Jalon V2-RC :** tag `v1.3.0`.

**Validation de fin de phase :** tous les points ci-dessus ; vérification explicite qu'aucun `DataFrame`, aucune ligne de données, n'apparaît dans les messages envoyés au LLM (test avec un `LLMProvider` factice qui enregistre ses entrées).

---

### Phase 28 — Optionnel : bayésien, qualité, plans d'expérience

**Cette phase est optionnelle.** Ne la démarre que si les phases 15 à 27 sont validées et qu'un besoin réel d'utilisateurs la justifie.
**Dépendances :** `uv add --optional bayes pymc arviz` ; `uv add --optional design pyDOE3` ; R : `qcc` (références).

**Tâches :**

- **T28.1 — Régression bayésienne simple** (`app/bayes/regression.py`) : `run_bayesian_linear_regression(df, outcome, predictors, prior_scale=..., draws=1000, chains=4, random_state=0)` avec **priors par défaut faiblement informatifs, affichés à l'utilisateur** (aucun prior choisi par un LLM, T14). Diagnostics : R-hat, ESS, divergences ; `NONCONVERGENCE` si R-hat > 1,01 (source : Vehtari et al. 2021, documenter) ; posterior predictive check. Limiter à de petits modèles (≤ 20 prédicteurs, ≤ 50 000 lignes) ; CPU-only : estimer et afficher le temps avant l'exécution.
- **T28.2 — Cartes de contrôle** (`app/quality/control_charts.py`) : `run_xbar_r_chart`, `run_p_chart`, `run_c_chart`, `compute_process_capability` (Cp, Cpk) ; validation contre `qcc`.
- **T28.3 — Plans factoriels et surfaces de réponse** (`app/quality/doe.py`) : `generate_factorial_design(factors, levels, fractional=None)` (`pyDOE3`), `analyze_factorial_design(df, response, factors)` (OLS + ANOVA), `fit_response_surface(df, response, factors)`.
- **T28.4 — Tests** (références : `FrF2`/`rsm` en R si disponibles, sinon valeurs analytiques documentées).

**Reportés (V3) :** facteurs de Bayes généraux, inférence variationnelle, réseaux bayésiens, plans optimaux (D-optimal), Taguchi, split-plot.

---

### Phase 29 — Stabilisation et release v2.0.0

**Objectif :** fiabiliser, mesurer les performances, documenter, tester avec de vrais utilisateurs. **Aucune nouvelle fonctionnalité** pendant cette phase.

**Tâches :**

- **T29.1 — Benchmarks de performance** (`tests/benchmarks/performance/`) : mesurer temps et mémoire pour 100 000 et 1 000 000 de lignes sur les méthodes principales de chaque famille ; consigner les limites dans la documentation et, si une méthode dépasse 4 Go (T22), ajouter l'échantillonnage manquant.
- **T29.2 — Audit des faux positifs** : exécuter tous les diagnostics V2 sur 10 datasets « propres » ; chaque faux positif conduit à un correctif de seuil documenté (source) ou à une rétrogradation de sévérité.
- **T29.3 — Documentation** : générer automatiquement `docs/methods.md` depuis le registre (`list_methods()` : méthode, famille, hypothèses, référence Python/R, extras) ; compléter `README.md` (installation par extras, limites connues, périmètre exclu V2 §1.3) et `CHANGELOG.md`.
- **T29.4 — Tests utilisateurs V2 (pas de code)** : 5 à 15 testeurs, formulaire structuré, backlog GitHub Issues priorisé ; critère : ≥ 5 testeurs terminent une analyse de leur choix sans aide.
- **T29.5 — Tag v2.0.0 (pas de code).**

**Validation de fin de phase :** CI verte (avec et sans R), `docs/methods.md` généré, rapport de validation contre R à jour pour toutes les méthodes, aucun défaut bloquant ouvert.

---

## 10. Annexe — Commandes utiles récapitulatives

```bash
# Environnement
uv sync --all-extras --dev          # installe tout, extras compris
uv sync --dev                       # installe le noyau seulement
uv add --optional <extra> <paquet>  # ajoute une dépendance à un extra

# Qualité
uv run pytest                       # tous les tests
uv run pytest -m "not slow"         # sans les tests lents
uv run pytest -m r_required         # uniquement les tests nécessitant R
uv run pytest tests/test_survival/  # une famille
uv run ruff check . && uv run ruff format .
uv run mypy app/

# Vérifier l'API d'une bibliothèque avant de coder
uv run python -c "import lifelines; help(lifelines.CoxPHFitter)"

# Registre
uv run python -c "from app.statistics.registry import list_methods; print(len(list_methods()))"

# R
Rscript app/r_bridge/scripts/<script>.R input.json output.json

# Jalons Git
git tag v1.1.0 && git tag v1.2.0 && git tag v1.3.0 && git tag v2.0.0 && git push --tags
```

---

## 11. Annexe — Checklist de validation globale V2

- ✅ Phase 15 : registre en place, méthodes V1 enregistrées, rejeu d'un `TransformLog`, import/export sur 6 formats, 1 M de lignes sous 4 Go
- ✅ Phase 16 : rapport EDA sur 3 datasets, concordance R à 1e-6
- [ ] Phase 17 : tests et tailles d'effet validés, 3 diagnostics d'inférence sans faux positif
- [ ] Phase 18 : GLM, ordinale, régularisée, robuste validés ; diagnostics de régression déclenchés sur cas construits
- [ ] Phase 19 : mixtes, GLMM, GEE validés contre R — **jalon V2-alpha (v1.1.0)**
- [ ] Phase 20 : ACP, AFE, AFC/ACM, clustering, LDA/QDA, SEM validés ; avertissements T24 présents
- [ ] Phase 21 : KM, log-rank, Cox, AFT, risques concurrents, fiabilité validés
- [ ] Phase 22 : séries, panel, IV validés ; backtesting sans fuite temporelle
- [ ] Phase 23 : enquêtes à plan complexe et méta-analyse validées contre R — **jalon V2-beta (v1.2.0)**
- [ ] Phase 24 : méthodes causales retrouvent l'effet simulé ; hypothèses d'identification affichées
- [ ] Phase 25 : pipelines ML reproductibles, sans fuite, interprétabilité échantillonnée
- [ ] Phase 26 : Word, LaTeX, PDF et notebook produits pour 5 méthodes
- [ ] Phase 27 : planner ≥ 75 % sur 40 questions, schémas générés pour chaque méthode, aucun microdonnée envoyée au LLM — **jalon V2-RC (v1.3.0)**
- [ ] Phase 28 (optionnelle) : bayésien, qualité, plans d'expérience
- [ ] Phase 29 : performances mesurées, audit de faux positifs, documentation générée, ≥ 5 testeurs externes, **v2.0.0 taguée**

*Fin du document. Contexte permanent à recoller à chaque session Qwen : V1 §2, V1 §6, V1 §7, V2 §3.2, V2 §7.*
