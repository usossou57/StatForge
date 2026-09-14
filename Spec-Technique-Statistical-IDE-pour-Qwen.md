**Spécification technique — Statistical IDE**  
**Document de contexte pour génération de code assistée (Qwen Chat / Qwen Coder)**  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OYQ1AABSAwc8mi5wvlAB6CKCAACr4Z7a7BLfMzFYdAQDwF+da3dX+9QQAgNeuB6fWBdZMUxZ2AAAAAElFTkSuQmCC)  
**Sommaire**  
1. Comment utiliser ce document (0.1 pour toi, 0.2 pour Qwen Chat)  
2. Contexte du projet  
3. Règle architecturale non négociable  
4. Contraintes techniques à respecter dans tout code généré  
5. Stack technique et installation  
6. Organisation du projet (arborescence cible)  
7. Conventions de code  
8. Contrats de données et interfaces centrales  
9. Gabarit de prompt pour Qwen  
10. Guide d'implémentation par phase (9.0 table de suivi + phases 0 à 14)  
11. Annexe — commandes utiles  
12. Annexe — checklist de validation globale  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAALUlEQVR4nO3OQQ0AIAwEsAMlSJ0UrOFkGngRklZBR1WtJDsAAPzizNcDAADuNcKwAyU+nb+5AAAAAElFTkSuQmCC)  
**0. Comment utiliser ce document**  
Ce document tient un double rôle : c'est une note d'usage pour toi (0.1), et un mode d'emploi que Qwen Chat doit lui-même suivre puisque ce fichier vit dans les sources de ton projet Qwen (0.2). Les deux sous-sections restent dans le document — ne retire pas 0.2 en pensant qu'elle "n'est pas pour toi", c'est justement celle que Qwen Chat doit lire et appliquer.  
**0.1 Pour toi (l'humain)**  
Ce n'est pas une feuille de route stratégique : c'est une **spécification technique**, déposée telle quelle dans les fichiers sources de ton projet Qwen Chat. Une fois qu'elle y est, tu n'as normalement plus besoin d'assembler toi-même des extraits de sections — c'est le travail de Qwen Chat (section 0.2).  
**Ton workflow :**  
1. Dépose ce fichier .md dans les fichiers sources du projet Qwen Chat.  
2. Pour démarrer une session, demande directement : *« où en est-on ? »* ou  *« par quoi on commence ? »*.  
3. Une fois la tâche identifiée, demande : *« donne-moi le prompt à soumettre à Qwen Coder pour cette tâche »*.  
4. Colle ce prompt dans Qwen Coder, récupère le code.  
5. Place-le exactement au chemin de fichier indiqué (section 9), lance les tests indiqués.  
6. Reviens dire à Qwen Chat que la tâche est faite, et mets à jour toi-même la case correspondante (⬜→✅) dans **ta** copie locale du document, section 9.0 — Qwen Chat ne peut pas modifier le fichier qu'il a reçu en source, il peut seulement en discuter.  
7. Si Qwen Chat ou Qwen Coder propose de modifier un contrat de la section 7 (signature de Variable, AnalysisResult, etc.), traite ça comme un signal d'alerte, jamais comme une suggestion à accepter automatiquement.  
**Une limite à vérifier toi-même :** je ne connais pas avec certitude le comportement exact de la mémoire de projet de Qwen Chat (context complet vs récupération partielle par extraits, persistance ou non d'une conversation à l'autre). Par prudence : garde un seul fil de conversation continu par période de travail si possible, et sinon rappelle en une phrase où tu en es (« on est à la tâche T4.2 ») avant de demander la suite — la table de suivi (section 9.0) sert de référence commune pour ça, qu'elle soit vue en entier par Qwen Chat ou seulement par extraits.  
**0.2 Pour Qwen Chat**  
*(mode d'emploi que Qwen Chat doit suivre — cette sous-section reste dans les fichiers du projet)*  
Tu es l'**orchestrateur** de ce projet, pas le rédacteur de code. Ton rôle se limite à deux choses :  
1. **Dire quoi faire ensuite.** Quand on te demande "par quoi on commence" ou "quelle est la prochaine étape", consulte la table de suivi (section 9.0) : identifie la première tâche non cochée (⬜), présente-la (ID, phase, fichier concerné) et résume sa spécification à partir de la section 9 correspondante. Si l'état d'avancement ne t'a pas été communiqué dans cette conversation, demande-le avant de répondre plutôt que de le deviner.  
2. **Composer le prompt pour Qwen Coder.** Quand on te demande le prompt pour une tâche donnée (par son ID, ex. "T3.4", ou par description), assemble-le en suivant EXACTEMENT le gabarit de la section 8 : contexte permanent (sections 2, 6, 7, recopiées intégralement — ce sont des contrats, ne les résume jamais) + contexte de la phase concernée (section 9) + la tâche précise (signature, spécification, chemin des tests). Livre le prompt entier, prêt à copier-coller tel quel dans Qwen Coder.  
**Ce que tu ne dois jamais faire :**  
- Écrire toi-même l'implémentation du code — ce travail revient à Qwen Coder.  
- Modifier une signature de la section 7 sans que l'utilisateur ne le demande explicitement.  
- Sauter une tâche ou réordonner les phases parce que ça semble plus rapide — l'ordre existe pour que le moteur statistique soit validé avant l'arrivée de la couche IA (section 2). Si l'utilisateur veut réordonner, c'est à lui de le demander explicitement.  
- Répondre "où en est-on" sans t'appuyer sur la table de suivi (section 9.0) — ne devine jamais la progression à partir du ton de la conversation.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhZscaUpheJwqQgQU2QtIq6DIze3UGAMBf3Gu1VcfXEwAAXrseopcEQ2uoYnwAAAAASUVORK5CYII=)  
**1. Contexte du projet**  
*(à donner à Qwen une seule fois, en tout début de première session)*  
Le projet est un **Statistical IDE** : un espace de travail statistique assisté par IA, à destination de chercheurs/étudiants/statisticiens individuels. L'utilisateur importe un dataset (CSV/XLSX/SAV), le système le profile automatiquement, l'utilisateur pose une question en langage naturel, le système propose un plan d'analyse, exécute réellement le calcul, détecte les problèmes statistiques (diagnostics) et explique les résultats en langage clair.  
Contraintes de développement : développeur solo, budget quasi nul, machine CPU-only (~12 Go RAM, pas de GPU), Linux Mint, développement local-first. Python est le langage principal ; R est un moteur complémentaire introduit plus tard (Phase 6).  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNBCkLfE07YGfHAiAU2QtIq6DIzW7UHAMBfnGt1V8fXEwAAXrse4eQF6VhvmPsAAAAASUVORK5CYII=)  
**2. Règle architecturale non négociable**  
*(à coller à Qwen dans CHAQUE session, sans exception)*  
   L'IA PROPOSE  →  LE MOTEUR CALCULE  →  LE VALIDATEUR CONTRÔLE  →  L'UTILISATEUR DÉCIDE  
   
**Règles dérivées, à faire respecter dans tout code généré :**  
- Un LLM ne calcule **jamais** directement une statistique (moyenne, coefficient, p-value). Il ne fait qu'appeler des fonctions Python déterministes et afficher/expliquer leur résultat.  
- Toute fonction de calcul statistique doit être testable et testée **indépendamment** de toute clé API — la suite de tests de app/data, app/statistics, app/diagnostics doit passer avec zéro configuration LLM.  
- Aucune fonction des modules app/statistics, app/diagnostics, app/results, app/data ne doit importer quoi que ce soit lié à un LLM (pas d'import de app.agent.* dans ces modules).  
- Un diagnostic (DiagnosticFinding) doit toujours être calculé par du code déterministe, jamais "halluciné" ou paraphrasé de mémoire par un LLM.  
- Le module app.agent (LLM) ne reçoit jamais de DataFrame brut ni de microdonnées individuelles — uniquement des dictionnaires de données (métadonnées) et des objets AnalysisResult déjà calculés.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OQQmAABRAsSd4NIGBzPXBmAawhhW8ibAl2DIze3UGAMBf3Gu1VcfXEwAAXrsehaQEN+8fLHEAAAAASUVORK5CYII=)  
**3. Contraintes techniques à respecter dans tout code généré**  
| | | |  
|-|-|-|  
| **#** | **Contrainte** | **Directive pour le code généré** |   
| T1 | Développeur solo | Code simple et lisible > code "clever" ; pas d'abstraction sans un deuxième cas d'usage concret |   
| T2 | Budget quasi nul | N'utiliser que des bibliothèques listées en section 4 — signaler explicitement toute proposition de dépendance payante |   
| T3 | CPU-only, ~12 Go RAM | Pas de chargement de modèle ML lourd en mémoire ; pandas par défaut, jamais Polars/DuckDB sauf demande explicite |   
| T4 | Local-first | Aucun appel réseau dans app/data, app/statistics, app/diagnostics, app/results |   
| T5 | Python 3.11+, typage strict | Toute fonction publique a des type hints complets et une docstring |   
| T6 | R complémentaire, pas prioritaire | N'implémenter le pont R (app/r_bridge) qu'à partir de la Phase 6, jamais avant |   
| T7 | Ne pas réinventer les maths | Toujours utiliser statsmodels/scipy pour un calcul statistique existant, jamais une implémentation "maison" |   
| T8 | Séparation calcul/IA stricte | Voir section 2 |   
| T9 | LLM provider-agnostic | Toute interaction LLM passe par le Protocol LLMProvider (section 7), jamais un appel API en dur ailleurs |   
| T10 | Architecture simple | Un seul déployable (modular monolith) ; pas de microservices, pas de queue, pas de base de données externe avant la Phase 11 |   
| T11 | Priorité : exactitude > confidentialité > reproductibilité > fiabilité > simplicité > UX > performance | En cas de doute entre deux implémentations possibles, appliquer cet ordre |   
| T12 | Validation contre R | Toute nouvelle méthode statistique doit être accompagnée d'un test comparant son résultat à une sortie R de référence (section 9, gabarit de tests) |   
| T13 | Reproductibilité | Toute fonction d'analyse doit retourner le code Python généré permettant de reproduire son propre résultat |   
| T14 | Confirmation humaine | Toute fonction de app/agent qui modifie un état (relance une analyse, applique une correction) doit exposer un paramètre explicite de confirmation, jamais s'exécuter silencieusement |   
| T15 | Sécurité progressive | Pas de eval()/exec() sur du texte généré par un LLM avant la Phase 13 (sandboxing) — si une tâche semble en avoir besoin avant, le signaler |   
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AUBBAsUfyVTCg9UygEBVsWGAjJK2CbjNzVGcAAPzFtapV7V9PAAB47X4AEWIEM8iQs0EAAAAASUVORK5CYII=)  
**4. Stack technique et installation**  
**4.1 Prérequis système (Linux Mint / Ubuntu-based)**  
# Python (si non présent) et outils de base  
 sudo apt update  
 sudo apt install -y python3 python3-dev git build-essential  
   
 # uv (gestionnaire de projet/dépendances Python)  
 curl -LsSf https://astral.sh/uv/install.sh | sh  
 # redémarrer le terminal, puis vérifier :  
 uv --version  
   
R n'est nécessaire qu'à partir de la **Phase 6**. Ne pas l'installer avant :  
sudo apt install -y r-base r-base-dev  
 Rscript -e 'install.packages(c("jsonlite", "logistf"), repos="https://cloud.r-project.org")'  
   
**4.2 Initialisation du projet (Phase 1 uniquement)**  
mkdir statistical-ai && cd statistical-ai  
 uv init --package --name statistical_ai --python 3.11  
 git init  
   
**4.3 Dépendances par phase (commandes exactes)**  
*Ne jamais ajouter une dépendance d'une phase ultérieure en avance. * *uv add* * résout et verrouille automatiquement les versions compatibles dans * *uv.lock* * — ne pas fixer de numéro de version à la main sauf raison précise documentée.*  
| | |  
|-|-|  
| **Phase** | **Commande** |   
| 1 — Fondations | uv add --dev pytest ruff mypy pre-commit |   
| 2 — Data Engine | uv add pandas numpy pyreadstat |   
| 3 — Statistical Engine V1 | uv add scipy statsmodels |   
| 4 — Result Architecture | uv add jinja2 matplotlib |   
| 5 — Diagnostic Engine V1 | *(aucune nouvelle dépendance — tout est déjà dans statsmodels)* |   
| 6 — Statistical Engine V2 | uv add firthlogist*(+ installation R, section 4.1)* |   
| 7 — AI Layer | uv add pydantic httpx*(+ le SDK du fournisseur LLM choisi, ex. * *uv add anthropic* * ou * *uv add openai* * selon le fournisseur retenu)* |   
| 8 — Analysis Planner | *(aucune nouvelle dépendance)* |   
| 9 — Statistical Agent | *(aucune nouvelle dépendance)* |   
| 10 — UI | uv add streamlit plotly |   
| 11 — Reproductibilité | *(aucune nouvelle dépendance — * *json* */* *sqlite3* * sont dans la stdlib)* |   
| 13 — Sécurité | *(aucune nouvelle dépendance — module * *resource* * et * *ast* * sont dans la stdlib)* |   
   
**4.4 **pyproject.toml ** — sections à faire vérifier par Qwen après chaque **uv add  
[tool.ruff]  
 line-length = 100  
 target-version = "py311"  
   
 [tool.ruff.lint]  
 select = ["E", "F", "I", "UP", "B"]  
   
 [tool.mypy]  
 python_version = "3.11"  
 disallow_untyped_defs = true  
 warn_return_any = true  
   
 [tool.pytest.ini_options]  
 testpaths = ["tests"]  
   
**4.5 **.pre-commit-config.yaml ** (Phase 1)**  
repos:  
   - repo: https://github.com/astral-sh/ruff-pre-commit  
     rev: v0.6.0  
     hooks:  
       - id: ruff  
       - id: ruff-format  
   - repo: local  
     hooks:  
       - id: pytest-check  
         name: pytest  
         entry: uv run pytest  
         language: system  
         pass_filenames: false  
         always_run: true  
   
*(Vérifier/mettre à jour le * *rev* * de * *ruff-pre-commit* * au moment de l'installation — indiqué ici à titre d'exemple de structure, pas comme version figée.)*  
**4.6 CI GitHub Actions minimale (**.github/workflows/ci.yml **, Phase 1-2)**  
name: CI  
 on: [push, pull_request]  
 jobs:  
   test:  
     runs-on: ubuntu-latest  
     steps:  
       - uses: actions/checkout@v4  
       - uses: astral-sh/setup-uv@v3  
       - run: uv sync --all-extras --dev  
       - run: uv run ruff check .  
       - run: uv run pytest  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhZscYahheJwqQgQU2QtIq6DIze3UGAMBf3Gu1VcfXEwAAXrseoqcEQXyAWBgAAAAASUVORK5CYII=)  
**5. Organisation du projet (arborescence cible)**  
*(Arborescence finale visée fin de Phase 11 — chaque tâche de la section 9 indique quels fichiers créer à quelle phase ; ne pas créer un dossier avant que sa première tâche n'arrive.)*  
statistical-ai/  
 ├── app/  
 │   ├── __init__.py  
 │   ├── data/  
 │   │   ├── __init__.py  
 │   │   ├── models.py            # Variable, DataDictionary (Phase 2)  
 │   │   ├── importers.py         # import CSV/XLSX/SAV (Phase 2)  
 │   │   └── profiling.py         # profile_dataset(), describe_variable() (Phase 2)  
 │   ├── statistics/  
 │   │   ├── __init__.py  
 │   │   ├── provider.py          # Protocol StatisticalEngineProvider (Phase 3)  
 │   │   ├── descriptive.py       # statistiques descriptives, tableaux croisés (Phase 3)  
 │   │   ├── tests_classiques.py  # run_chi_square, run_t_test, run_anova (Phase 3)  
 │   │   ├── regression.py        # OLS, Logit (Phase 3), MNLogit, Firth (Phase 6)  
 │   │   └── engine_python.py     # implémentation Python de StatisticalEngineProvider (Phase 3)  
 │   ├── diagnostics/  
 │   │   ├── __init__.py  
 │   │   ├── models.py            # DiagnosticFinding, Severity (Phase 5)  
 │   │   └── checks.py            # check_multicollinearity, check_separation, influence (Phase 5)  
 │   ├── results/  
 │   │   ├── __init__.py  
 │   │   ├── models.py            # AnalysisResult (Phase 4)  
 │   │   └── codegen.py           # génération du code Python reproductible (Phase 4)  
 │   ├── planner/  
 │   │   ├── __init__.py  
 │   │   └── models.py            # AnalysisPlan (Phase 8)  
 │   │   └── planner.py           # logique du planner (Phase 8)  
 │   ├── agent/  
 │   │   ├── __init__.py  
 │   │   ├── llm_provider.py      # Protocol LLMProvider + implémentations (Phase 7)  
 │   │   ├── tools.py             # wrappers exposés au tool calling (Phase 7)  
 │   │   └── orchestrator.py      # agent confirm-then-execute (Phase 9)  
 │   ├── reports/  
 │   │   ├── __init__.py  
 │   │   ├── generator.py         # export Markdown/HTML (Phase 4), PDF (Phase 11+)  
 │   │   └── templates/           # templates Jinja2 (Phase 4)  
 │   ├── r_bridge/  
 │   │   ├── __init__.py  
 │   │   ├── bridge.py            # call_r_script() (Phase 6)  
 │   │   └── scripts/             # scripts .R (Phase 6)  
 │   │       └── firth_logistic.R  
 │   ├── reproducibility/  
 │   │   ├── __init__.py  
 │   │   ├── project.py           # Project, persistence (Phase 11)  
 │   │   └── history.py           # journal des analyses (Phase 11)  
 │   ├── security/  
 │   │   ├── __init__.py  
 │   │   └── sandbox.py           # exécution isolée, limites de ressources (Phase 13)  
 │   └── ui/  
 │       └── streamlit_app.py     # interface Streamlit (Phase 10)  
 ├── tests/  
 │   ├── fixtures/                # datasets de test (Phase 2+)  
 │   ├── benchmarks/               # datasets de référence + résultats R attendus (Phase 3+)  
 │   ├── test_data/  
 │   ├── test_statistics/  
 │   ├── test_diagnostics/  
 │   ├── test_results/  
 │   ├── test_planner/  
 │   └── test_agent/  
 ├── docs/  
 │   └── adr/  
 │       └── ADR-000-template.md  
 ├── examples/  
 ├── scripts/  
 ├── pyproject.toml  
 ├── uv.lock  
 ├── README.md  
 ├── CHANGELOG.md  
 └── .pre-commit-config.yaml  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OMQ0AIAwAwZIgBKn1gjJsdGLBABMhuZt+/JaZIyJmAADwi9VP1NMNAABu1AaU4gUeBSGW2wAAAABJRU5ErkJggg==)  
**6. Conventions de code**  
*(à coller à Qwen dans CHAQUE session, sans exception)*  
- **Langue :** identifiants (noms de fonctions, classes, variables) en  **anglais**, conforme aux conventions Python/OSS. Docstrings, commentaires, et tout message destiné à l'utilisateur final (erreurs, diagnostics) en  **français**.  
- **Style :** PEP 8, imposé par Ruff (uv run ruff check . doit passer sans erreur). Longueur de ligne max 100 caractères.  
- **Typage :** type hints complets et obligatoires sur toute fonction publique (paramètres + retour). mypy doit passer sans erreur sur app/.  
- **Docstrings :** format Google, en français :  
- def profile_dataset(df: pd.DataFrame) -> dict:  
     """Calcule un profil descriptif complet d'un DataFrame.  
   
     Args:  
         df: Le DataFrame à profiler.  
   
     Returns:  
         Un dictionnaire contenant dimensions, types, valeurs manquantes,  
         cardinalité et variables constantes détectées.  
     """  
   
- **Tests :** un fichier de test par module source (app/data/profiling.py → tests/test_data/test_profiling.py). Nommage def test_<comportement_testé>():. Chaque fonction statistique publique doit avoir au minimum : un test de cas nominal, un test de cas limite (dataset vide, valeur manquante, catégorie unique), et — dès qu'une méthode statistique est concernée — un test de validation numérique contre une valeur de référence R (voir gabarit ci-dessous).  
- **Gabarit de test de validation numérique (à demander explicitement à Qwen pour toute nouvelle méthode) :**  
- def test_linear_regression_matches_r_reference():  
     """Compare le coefficient obtenu à la sortie R de référence (tolérance 1e-6)."""  
     result = run_linear_regression(df, outcome="y", predictors=["x1", "x2"])  
     assert result.estimates["x1"] == pytest.approx(R_REFERENCE_COEF_X1, rel=1e-6)  
   
- **Commits :**[Conventional Commits — feat(data): ajoute profile_dataset, test(statistics): ajoute validation OLS vs R, etc.](https://www.conventionalcommits.org/ "https://www.conventionalcommits.org/")  
- **Dépendances :** ne jamais introduire une bibliothèque hors de la section 4.3 sans le signaler explicitement en réponse ("cette implémentation nécessite d'ajouter X, à valider avant").  
- **Pas de code mort ni de "TODO" silencieux :** si une partie n'est pas implémentée, lever NotImplementedError avec un message clair plutôt que de laisser une fonction vide.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OUQmAABBAsSdYxKbXxlpGEAOIFfwTYUuwZWa2ag8AgL841uquzq8nAAC8dj05VAYO3phhoQAAAABJRU5ErkJggg==)  
**7. Contrats de données et interfaces centrales**  
*(à coller à Qwen dans CHAQUE session — ce sont les contrats que tout le code généré doit respecter sans exception ; ne jamais laisser Qwen les modifier sans validation explicite de ta part)*  
# app/data/models.py  
 from dataclasses import dataclass, field  
   
   
 @dataclass  
 class Variable:  
     """Représentation interne d'une variable d'un dataset."""  
     name: str  
     label: str | None = None  
     dtype: str = "unknown"                 # "numeric" | "categorical" | "datetime" | "text"  
     semantic_type: str | None = None       # "binary" | "ordinal" | "count" | "continuous"  
     levels: list[str] | None = None  
     labels: dict[str, str] | None = None  
     missing_values: list = field(default_factory=list)  
     role: str | None = None                # "outcome" | "predictor" | "id" | "weight"  
     reference_category: str | None = None  
     weight: str | None = None  
     metadata: dict = field(default_factory=dict)  
   
   
 @dataclass  
 class DataDictionary:  
     """Dictionnaire de données interne d'un dataset importé."""  
     dataset_name: str  
     n_rows: int  
     n_columns: int  
     variables: dict[str, Variable]  
     source_path: str  
     file_hash: str  
   
     def to_json(self) -> str: ...  
     @classmethod  
     def from_json(cls, json_str: str) -> "DataDictionary": ...  
   
# app/diagnostics/models.py  
 from dataclasses import dataclass, field  
 from enum import Enum  
   
   
 class Severity(str, Enum):  
     """Niveau de gravité d'un diagnostic — ne jamais transformer un warning en blocage."""  
     CERTAIN_ISSUE = "certain_issue"  
     PROBABLE_ISSUE = "probable_issue"  
     WARNING = "warning"  
     INFORMATION = "information"  
   
   
 @dataclass  
 class DiagnosticFinding:  
     """Un diagnostic détecté par le Diagnostic Engine — toujours produit par du code  
     déterministe, jamais par un LLM."""  
     code: str                      # ex. "SEPARATION_COMPLETE", "HIGH_VIF"  
     severity: Severity  
     message: str                   # explication en langage clair (français)  
     details: dict = field(default_factory=dict)  
   
# app/results/models.py  
 from dataclasses import dataclass, field  
 from datetime import datetime, timezone  
   
 from app.diagnostics.models import DiagnosticFinding  
   
   
 @dataclass  
 class AnalysisResult:  
     """Objet structuré unique représentant le résultat d'une analyse — jamais  
     remplacé par un simple tableau affiché à l'écran."""  
     method: str  
     dataset_id: str  
     variables: dict  
     sample_size: int  
     preprocessing: list[str]  
     estimates: dict  
     standard_errors: dict  
     confidence_intervals: dict  
     p_values: dict  
     model_metrics: dict  
     diagnostics: list[DiagnosticFinding]  
     warnings: list[str]  
     assumptions_checked: list[str]  
     generated_code: str  
     execution_log: list[str]  
     software_versions: dict  
     created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  
   
     def to_json(self) -> str: ...  
   
# app/statistics/provider.py  
 from typing import Protocol  
 import pandas as pd  
   
 from app.results.models import AnalysisResult  
   
   
 class StatisticalEngineProvider(Protocol):  
     """Abstraction permettant, à terme, un moteur Python et un moteur R  
     interchangeables (C7/T6). Ne jamais appeler statsmodels/R directement  
     en dehors d'une implémentation de ce Protocol."""  
   
     def run_linear_regression(  
         self, df: pd.DataFrame, outcome: str, predictors: list[str]  
     ) -> AnalysisResult: ...  
   
     def run_logistic_regression(  
         self, df: pd.DataFrame, outcome: str, predictors: list[str],  
         reference_category: str | None = None  
     ) -> AnalysisResult: ...  
   
     # Complétée au fil des phases : run_chi_square, run_t_test, run_anova,  
     # run_multinomial_logistic_regression, run_firth_logistic, compare_models.  
   
# app/agent/llm_provider.py  
 from typing import Protocol, Any  
   
   
 class LLMProvider(Protocol):  
     """Abstraction agnostique du fournisseur LLM (C10/T9). Toute interaction  
     avec un modèle de langage passe par cette interface, nulle part ailleurs."""  
   
     def complete(  
         self,  
         messages: list[dict[str, str]],  
         tools: list[dict[str, Any]] | None = None,  
         response_schema: type | None = None,  
     ) -> "LLMResponse": ...  
   
   
 class LLMResponse:  
     """Résultat d'un appel LLM : texte libre, appel(s) d'outil, ou objet structuré."""  
     text: str | None  
     tool_calls: list[dict[str, Any]] | None  
     structured_output: Any | None  
   
# app/planner/models.py  
 from typing import Literal  
 from pydantic import BaseModel  
   
   
 class AnalysisPlan(BaseModel):  
     """Plan d'analyse structuré produit par l'Analysis Planner — jamais du  
     texte libre, toujours validé par ce schéma Pydantic."""  
     research_question: str  
     outcome_variable: str  
     predictor_variables: list[str]  
     study_design: Literal["cross_sectional", "comparative", "associative", "predictive"]  
     proposed_method: str  
     reference_category: str | None = None  
     assumptions_to_check: list[str]  
     rationale: str  
     confidence: Literal["high", "medium", "low"]  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAM0lEQVR4nO3OUQmAQBBAwSdcjsu6HYxoDsEK/okwk2COmdnVGQAAf3GtalX76wkAAK/dDxFWBDkFf6+SAAAAAElFTkSuQmCC)  
**8. Gabarit de prompt pour Qwen**  
Utilise ce gabarit pour **chaque** tâche, en le complétant avec la tâche exacte tirée de la section 9. Ne saute jamais la partie "Contexte permanent".  
### Contexte permanent (colle ici les sections 2, 6 et 7 de la spec technique)  
   
 ### Contexte de phase (colle ici la sous-section de la Phase en cours, section 9)  
   
 ### Tâche  
   
 Implémente uniquement ce qui suit, rien de plus :  
   
 Fichier : <chemin exact, ex. app/data/profiling.py>  
   
 Signature à respecter exactement :  
 def profile_dataset(df: pd.DataFrame) -> dict:  
     """..."""  
   
 Spécification :  
 - <détails fonctionnels tirés de la section 9>  
   
 Contraintes :  
 - Respecte exactement la signature ci-dessus, ne change aucun contrat de la section 7  
 - N'utilise que les dépendances déjà installées à cette phase (section 4.3)  
 - Fournis aussi le(s) test(s) pytest correspondant(s) dans <chemin exact du fichier de test>  
 - Si tu penses qu'une dépendance ou un contrat doit changer, signale-le au lieu de le faire silencieusement  
   
**Exemple concret rempli (Phase 2, une tâche) :**  
### Tâche  
   
 Fichier : app/data/profiling.py  
   
 Signature à respecter exactement :  
 def profile_dataset(df: pd.DataFrame) -> dict:  
     """..."""  
   
 Spécification :  
 - Retourne un dict avec les clés : "n_rows", "n_columns", "dtypes" (dict colonne->type),  
   "missing_pct" (dict colonne->% de valeurs manquantes), "n_unique" (dict colonne->cardinalité),  
   "constant_columns" (liste des colonnes à valeur unique)  
 - Utilise uniquement pandas/numpy  
   
 Contraintes :  
 - Fournis aussi les tests dans tests/test_data/test_profiling.py, avec un cas nominal  
   et un cas limite (DataFrame avec une colonne constante et une colonne à 80% de valeurs manquantes)  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhYMMAKlD4OzrxgQU2QtIq6DIzR3UFAMBf3Gu1VefXEwAAXtsfSqADWz4G/HUAAAAASUVORK5CYII=)  
**9. Guide d'implémentation par phase**  
*Pour chaque phase : objectif technique, dépendances (déjà données section 4.3, rappelées si pertinent), fichiers à créer/modifier, spécification technique tâche par tâche, tests à écrire, validation de fin de phase. Une tâche = une requête à Qwen.*  
**9.0 Suivi d'avancement**  
*(Sourc* *e de vérité pour répondre à "où en est-on"/"par quoi on commence". Statut mis à jour manuellement par l'humain — section 0.1 — au fil de l'avancement. Chaque ID correspond, dans l'ordre, aux tâches numérotées de la sous-section de phase indiquée plus bas.)*  
| | | | | |  
|-|-|-|-|-|  
| **ID** | **Phase** | **Tâche (résumé)** | **Fichier principal** | **Statut** |   
| T0.1 | 0 — Cadrage | Rédiger docs/product-brief.md (pas de code, pas de Qwen) | docs/product-brief.md | ✅ |   
| T1.1 | 1 — Fondations | Config CI/pre-commit | .pre-commit-config.yaml, .github/workflows/ci.yml | ⬜ |   
| T1.2 | 1 — Fondations | Gabarit d'ADR | docs/adr/ADR-000-template.md | ⬜ |   
| T1.3 | 1 — Fondations | ADR-001 et ADR-002 (rédigées par toi, pas Qwen) | docs/adr/ADR-001-*.md, ADR-002-*.md | ⬜ |   
| T1.4 | 1 — Fondations | Test trivial de câblage CI | tests/test_setup.py | ⬜ |   
| T2.1 | 2 — Data Engine | Classes Variable, DataDictionary | app/data/models.py | ⬜ |   
| T2.2 | 2 — Data Engine | Import CSV/XLSX/SAV | app/data/importers.py | ⬜ |   
| T2.3 | 2 — Data Engine | Profiling + dictionnaire de données | app/data/profiling.py | ⬜ |   
| T2.4 | 2 — Data Engine | Tests + fixtures (dataset "sale") | tests/test_data/ | ⬜ |   
| T3.1 | 3 — Statistical Engine V1 | Protocol StatisticalEngineProvider | app/statistics/provider.py | ⬜ |   
| T3.2 | 3 — Statistical Engine V1 | Tableaux croisés | app/statistics/descriptive.py | ⬜ |   
| T3.3 | 3 — Statistical Engine V1 | χ², test t, ANOVA | app/statistics/tests_classiques.py | ⬜ |   
| T3.4 | 3 — Statistical Engine V1 | Régression OLS et logistique | app/statistics/regression.py | ⬜ |   
| T3.5 | 3 — Statistical Engine V1 | PythonStatisticalEngine | app/statistics/engine_python.py | ⬜ |   
| T3.6 | 3 — Statistical Engine V1 | Tests + validation numérique vs R | tests/test_statistics/ | ⬜ |   
| T4.1 | 4 — Result Architecture | AnalysisResult | app/results/models.py | ⬜ |   
| T4.2 | 4 — Result Architecture | Génération de code reproductible | app/results/codegen.py | ⬜ |   
| T4.3 | 4 — Result Architecture | Génération de rapports + graphiques | app/reports/generator.py | ⬜ |   
| T4.4 | 4 — Result Architecture | Template HTML | app/reports/templates/report.html.j2 | ⬜ |   
| T4.5 | 4 — Result Architecture | Tests | tests/test_results/ | ⬜ |   
| T5.1 | 5 — Diagnostic Engine V1 | Severity, DiagnosticFinding | app/diagnostics/models.py | ⬜ |   
| T5.2 | 5 — Diagnostic Engine V1 | VIF, séparation, influence, taille d'échantillon | app/diagnostics/checks.py | ⬜ |   
| T5.3 | 5 — Diagnostic Engine V1 | Corpus de cas limites + tests | tests/fixtures/edge_cases/ | ⬜ |   
| **— Jalon MVP —** |   | **Moteur + diagnostics validés, zéro IA** |   | ⬜ |   
| T6.1 | 6 — Statistical Engine V2 | MNLogit, compare_models | app/statistics/regression.py | ⬜ |   
| T6.2 | 6 — Statistical Engine V2 | Pont R (call_r_script) | app/r_bridge/bridge.py | ⬜ |   
| T6.3 | 6 — Statistical Engine V2 | Script R Firth | app/r_bridge/scripts/firth_logistic.R | ⬜ |   
| T6.4 | 6 — Statistical Engine V2 | run_firth_logistic | app/statistics/regression.py | ⬜ |   
| T6.5 | 6 — Statistical Engine V2 | Tests pont R + validation | tests/test_r_bridge/ | ⬜ |   
| T7.1 | 7 — AI Layer | Protocol LLMProvider + implémentations | app/agent/llm_provider.py | ⬜ |   
| T7.2 | 7 — AI Layer | TOOL_SCHEMAS, execute_tool | app/agent/tools.py | ⬜ |   
| T7.3 | 7 — AI Layer | Tests | tests/test_agent/test_tools.py | ⬜ |   
| T8.1 | 8 — Analysis Planner | AnalysisPlan | app/planner/models.py | ⬜ |   
| T8.2 | 8 — Analysis Planner | build_analysis_plan, needs_clarification | app/planner/planner.py | ⬜ |   
| T8.3 | 8 — Analysis Planner | Jeu de 20 questions + tests | tests/test_planner/ | ⬜ |   
| T9.1 | 9 — Statistical Agent | propose_correction, execute_confirmed_action | app/agent/orchestrator.py | ⬜ |   
| T9.2 | 9 — Statistical Agent | Tests scénario séparation→Firth | tests/test_agent/test_orchestrator.py | ⬜ |   
| T10.1 | 10 — UI | Page upload + dictionnaire de données | app/ui/streamlit_app.py | ⬜ |   
| T10.2 | 10 — UI | Zone de chat | app/ui/streamlit_app.py | ⬜ |   
| T10.3 | 10 — UI | Zone de résultats/export | app/ui/streamlit_app.py | ⬜ |   
| T11.1 | 11 — Reproductibilité v2 | save_analysis, replay_analysis | app/reproducibility/project.py | ⬜ |   
| T11.2 | 11 — Reproductibilité v2 | Journal d'historique | app/reproducibility/history.py | ⬜ |   
| T11.3 | 11 — Reproductibilité v2 | Tests de rejeu (10 analyses) | tests/test_reproducibility/ | ⬜ |   
| T12.1 | 12 — Tests utilisateurs | Recrutement + retours (pas de code) | — | ⬜ |   
| T13.1 | 13 — Sécurité | validate_generated_code, run_sandboxed | app/security/sandbox.py | ⬜ |   
| T13.2 | 13 — Sécurité | Tests scénarios malveillants | tests/test_security/ | ⬜ |   
| T14.1 | 14 — Beta | README/CHANGELOG/tag v1.0.0 (pas de code) | README.md, CHANGELOG.md | ⬜ |   
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OMQ2AABAAsSNBACP6MMH6NpGACyywEZJWQZeZ2aszAAD+4l6rrTq+ngAA8Nr1AL+6BElk4wV6AAAAAElFTkSuQmCC)  
**Phase 0 — Cadrage (pas de code)**  
Pas de tâche Qwen ici. Rédige docs/product-brief.md toi-même : 3 cas d'usage réels, liste des 10 méthodes statistiques visées au départ, 3-5 datasets de démarrage identifiés (dont au moins un avec valeurs manquantes et catégories rares).  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OYQ1AABSAwc8mi5wvlAB6CKCAACr4Z7a7BLfMzFYdAQDwF+da3dX+9QQAgNeuB6fWBdZMUxZ2AAAAAElFTkSuQmCC)  
**Phase 1 — Fondations**  
**Objectif technique :** dépôt fonctionnel, testable, avec CI.  
**Fichiers à créer :** pyproject.toml, app/__init__.py, tests/__init__.py, .pre-commit-config.yaml, .github/workflows/ci.yml, docs/adr/ADR-000-template.md, docs/adr/ADR-001-python-first.md, docs/adr/ADR-002-modular-monolith.md.  
**Tâches :**  
1. Demande à Qwen le contenu de .pre-commit-config.yaml et .github/workflows/ci.yml en partant des gabarits de la section 4.5/4.6, adaptés si besoin.  
2. Demande le gabarit docs/adr/ADR-000-template.md (sections : Contexte, Problème, Options, Décision, Conséquences, Condition de révision).  
3. Écris toi-même ADR-001 et ADR-002 (décisions déjà prises, pas besoin de Qwen).  
**Tests à écrire :** un test trivial (tests/test_setup.py::test_import_app) qui vérifie que import app fonctionne — sert uniquement à valider que la CI est correctement câblée.  
**Validation de fin de phase :** uv run pytest passe en local et en CI ; uv run ruff check . ne remonte aucune erreur ; les deux ADR existent.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OQQmAUBBAwSf8GGLWDWFDY3ixgjcRZhLMNjNHdQYAwF9cq1rV/vUEAIDX7gcRXAQ2s/16gwAAAABJRU5ErkJggg==)  
**Phase 2 — Data Engine**  
**Dépendances :** uv add pandas numpy pyreadstat  
**Fichiers à créer :** app/data/models.py, app/data/importers.py, app/data/profiling.py.  
**Tâches :**  
1. **app/data/models.py** — demande la classe Variable et DataDictionary exactement comme spécifiées section 7, plus les méthodes to_json/from_json (utiliser dataclasses.asdict + json.dumps/json.loads).  
2. **app/data/importers.py** — trois fonctions :  
3. def import_csv(path: str) -> pd.DataFrame: ...  
 def import_xlsx(path: str) -> pd.DataFrame: ...  
 def import_sav(path: str) -> tuple[pd.DataFrame, dict]:  
     """Retourne (df, labels) où labels contient les labels de valeurs  
     extraits via pyreadstat (meta.variable_value_labels)."""  
   
4. **app/data/profiling.py** — trois fonctions :  
5. def profile_dataset(df: pd.DataFrame) -> dict: ...  
 def describe_variable(df: pd.DataFrame, column: str) -> dict: ...  
 def build_data_dictionary(df: pd.DataFrame, dataset_name: str, source_path: str) -> "DataDictionary": ...  
   
6. profile_dataset : voir spécification donnée dans l'exemple de la section 8.  
 describe_variable : pour une variable numérique retourne mean/median/std/min/max/n_missing ; pour une catégorielle retourne les fréquences par modalité.  
 build_data_dictionary : assemble un Variable par colonne (dtype inféré via pd.api.types, semantic_type déduit par une heuristique simple : binaire si 2 modalités, ordinal/nominal sinon) en un DataDictionary complet, avec file_hash calculé via hashlib.sha256 sur le fichier source.  
**Tests à écrire :** tests/test_data/test_importers.py, tests/test_data/test_profiling.py — cas nominal + un dataset de test avec valeurs manquantes, une colonne constante, une catégorie rare (crée ces fixtures dans tests/fixtures/).  
**Validation de fin de phase :** python -m app.data.profiling <un vrai CSV> (ou script équivalent) produit un profiling correct sur 3 datasets réels différents, dont un avec des données "sales".  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAM0lEQVR4nO3OMQ0AIAwAwZKQ6kBqjSAOJywYYCIkd9OP36pqRMQMAAB+sfqJfLoBAMCN3NYoAzBA+QG0AAAAAElFTkSuQmCC)  
**Phase 3 — Statistical Engine V1**  
**Dépendances :** uv add scipy statsmodels  
**Fichiers à créer :** app/statistics/provider.py, app/statistics/descriptive.py, app/statistics/tests_classiques.py, app/statistics/regression.py, app/statistics/engine_python.py.  
**Tâches :**  
1. **app/statistics/provider.py** — le Protocol StatisticalEngineProvider exact de la section 7.  
2. **app/statistics/descriptive.py** :  
3. def create_contingency_table(df: pd.DataFrame, row: str, col: str) -> pd.DataFrame: ...  
   
4. **app/statistics/tests_classiques.py** — chacune retourne un AnalysisResult (section 7) :  
5. def run_chi_square(df: pd.DataFrame, var1: str, var2: str) -> AnalysisResult:  
     """Utilise scipy.stats.chi2_contingency."""  
 def run_t_test(df: pd.DataFrame, outcome: str, group: str) -> AnalysisResult:  
     """Utilise scipy.stats.ttest_ind. group doit avoir exactement 2 modalités  
     (lever ValueError sinon, avec message clair en français)."""  
 def run_anova(df: pd.DataFrame, outcome: str, group: str) -> AnalysisResult:  
     """Utilise statsmodels.formula.api.ols + statsmodels.stats.anova.anova_lm."""  
   
6. **app/statistics/regression.py** :  
7. def run_linear_regression(df: pd.DataFrame, outcome: str, predictors: list[str]) -> AnalysisResult:  
     """Utilise statsmodels.formula.api.ols. Construit la formule à partir de  
     outcome/predictors. Remplit estimates/standard_errors/confidence_intervals/  
     p_values à partir de model.params/bse/conf_int()/pvalues. generated_code  
     doit contenir le code Python exact permettant de reproduire ce résultat."""  
 def run_logistic_regression(df: pd.DataFrame, outcome: str, predictors: list[str],  
                              reference_category: str | None = None) -> AnalysisResult:  
     """Utilise statsmodels.api.Logit ou GLM(family=Binomial). Convertit les  
     odds ratios (exp(coef)) en plus des coefficients bruts dans model_metrics."""  
   
8. **app/statistics/engine_python.py** — classe PythonStatisticalEngine qui implémente le Protocol en déléguant aux fonctions ci-dessus.  
**Tests à écrire :** pour chaque méthode, un test de cas nominal, un test de cas limite (ex. groupe à une seule modalité pour run_t_test, dataset vide) et  **un test de validation numérique contre une référence R** (gabarit section 6) — calcule la référence toi-même en R une seule fois et fige-la en constante dans le test.  
**Validation de fin de phase :** les 5 méthodes (χ², t, ANOVA, OLS, logistique) sont validées à une tolérance documentée (1e-6 pour OLS vs lm(), tolérance plus large et justifiée pour la logistique si les algorithmes d'optimisation diffèrent) sur au moins 2 datasets réels chacune.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AUBBAsUfyNTCi9VwgEA3sWGAjJK2CbjNzVGcAAPzFtapV7V9PAAB47X4AEW4ELQDBN+AAAAAASUVORK5CYII=)  
**Phase 4 — Result Architecture**  
**Dépendances :** uv add jinja2 matplotlib  
**Fichiers à créer :** app/results/models.py, app/results/codegen.py, app/reports/generator.py, app/reports/templates/report.html.j2.  
**Tâches :**  
1. **app/results/models.py** — la dataclass AnalysisResult exacte de la section 7, avec to_json.  
2. **app/results/codegen.py** :  
3. def generate_reproducible_code(method: str, params: dict) -> str:  
     """Retourne un script Python autonome (imports inclus) qui, exécuté  
     tel quel sur le même fichier de données, reproduit l'analyse."""  
   
4. **app/reports/generator.py** :  
5. def generate_report(result: "AnalysisResult", output_format: str = "html") -> str:  
     """output_format: 'markdown' ou 'html'. Utilise Jinja2 pour 'html'.  
     Contenu obligatoire : méthode, variables, tableau de coefficients/IC/p-values,  
     diagnostics, code généré, versions logicielles."""  
 def generate_plot(result: "AnalysisResult", kind: str) -> "matplotlib.figure.Figure":  
     """kind: 'coefficients' (forest plot simple) ou 'residuals'."""  
   
6. Template Jinja2 report.html.j2 — structure HTML minimale reprenant les sections listées ci-dessus.  
**Tests à écrire :** tests/test_results/test_codegen.py (le code généré s'exécute sans erreur via exec() dans un test contrôlé — seul endroit du projet où exec() est acceptable, car sur du code que  **toi** tu as généré et relu, pas un LLM en production), tests/test_results/test_reports.py.  
**Validation de fin de phase :** pour chacune des 5 méthodes de la Phase 3, un rapport HTML complet est généré et le code reproductible s'exécute correctement de façon autonome.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OMQ2AABAAsSNBCUpfDq4wwIAABiywEZJWQZeZ2ao9AAD+4liruzq/ngAA8Nr1ABweBgdur/QFAAAAAElFTkSuQmCC)  
**Phase 5 — Diagnostic Engine V1**  
**Dépendances :** aucune nouvelle (déjà dans statsmodels).  
**Fichiers à créer :** app/diagnostics/models.py, app/diagnostics/checks.py.  
**Tâches :**  
1. **app/diagnostics/models.py** — Severity (Enum) et DiagnosticFinding exacts de la section 7.  
2. **app/diagnostics/checks.py** :  
3. def check_multicollinearity(df: pd.DataFrame, predictors: list[str]) -> list[DiagnosticFinding]:  
     """VIF via statsmodels.stats.outliers_influence.variance_inflation_factor.  
     Seuils : VIF > 10 -> CERTAIN_ISSUE, VIF > 5 -> PROBABLE_ISSUE, sinon rien.  
     Documenter ces seuils en commentaire avec leur source."""  
 def check_separation(model_result) -> list[DiagnosticFinding]:  
     """Détecte séparation complète/quasi-séparation sur un résultat de  
     régression logistique : erreurs standard anormalement grandes  
     (ex. > 10x le coefficient) et/ou non-convergence signalée par statsmodels."""  
 def check_influential_observations(model_result) -> list[DiagnosticFinding]:  
     """Distance de Cook via statsmodels.stats.outliers_influence.OLSInfluence  
     (ou GLMInfluence pour la logistique). Seuil : distance > 4/n -> WARNING."""  
 def check_sample_size(df: pd.DataFrame, method: str, n_predictors: int) -> list[DiagnosticFinding]:  
     """Règle empirique : au moins 10 événements par variable pour la  
     régression logistique (Events Per Variable, EPV)."""  
   
**Tests à écrire :** un corpus de cas limites dans tests/fixtures/edge_cases/ (séparation complète construite exprès, colinéarité parfaite construite exprès, catégorie vide) + un test par fonction sur chaque cas limite, vérifiant le bon code et la bonne Severity retournés.  
**Validation de fin de phase :** chaque diagnostic ci-dessus est déclenché correctement sur son cas limite dédié, et ne se déclenche PAS sur un dataset "propre" (test de non-régression du faux positif).  
***Jalon MVP : à la fin de cette phase, le moteur (Phases 2 à 5) doit fonctionner de bout en bout sans aucune dépendance IA.*** * Avant de passer à la Phase 6, vérifie que * *uv run pytest* * passe intégralement sans clé API configurée nulle part.*  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OQQmAABRAsSfYxKK/kYXEkyk8WcGbCFuCLTOzVXsAAPzFuVZ3dXw9AQDgtesB/v8F8JQadPwAAAAASUVORK5CYII=)  
**Phase 6 — Statistical Engine V2 (+ pont R)**  
**Dépendances :** uv add firthlogist + installation R (section 4.1)  
**Fichiers à créer :** app/statistics/regression.py (complété), app/r_bridge/bridge.py, app/r_bridge/scripts/firth_logistic.R.  
**Tâches :**  
1. **app/statistics/regression.py** (ajouts) :  
2. def run_multinomial_logistic_regression(df: pd.DataFrame, outcome: str,  
                                          predictors: list[str]) -> AnalysisResult:  
     """Utilise statsmodels.api.MNLogit."""  
 def compare_models(results: list[AnalysisResult]) -> dict:  
     """Compare AIC/BIC/log-likelihood entre plusieurs AnalysisResult  
     de même famille de méthode."""  
   
3. **app/r_bridge/bridge.py** :  
4. def call_r_script(script_name: str, payload: dict, timeout: int = 60) -> dict:  
     """Écrit payload en JSON dans un fichier temporaire, appelle  
     `Rscript app/r_bridge/scripts/<script_name>` via subprocess avec les  
     chemins des fichiers d'entrée/sortie en arguments, lit et retourne le  
     JSON de sortie. Lève une exception claire si R n'est pas installé  
     (dégradation propre, jamais un crash silencieux)."""  
   
5. **app/r_bridge/scripts/firth_logistic.R** — script R minimal utilisant logistf et jsonlite, lisant args[1] (JSON d'entrée : data, outcome, predictors) et écrivant args[2] (JSON de sortie : coefficients, standard_errors, p_values).  
6. **app/statistics/regression.py** (ajout) :  
7. def run_firth_logistic(df: pd.DataFrame, outcome: str, predictors: list[str]) -> AnalysisResult:  
     """Tente d'abord firthlogist (Python) ; si le résultat semble incohérent  
     ou si firthlogist échoue sur ce cas, utilise call_r_script('firth_logistic.R', ...)  
     comme repli, en le signalant dans warnings."""  
   
**Tests à écrire :** tests/test_r_bridge/test_bridge.py (test d'intégration marqué @pytest.mark.skipif si R n'est pas installé sur la machine de CI), validation numérique de run_firth_logistic contre logistf en R directement.  
**Validation de fin de phase :** run_firth_logistic fonctionne et est validé contre R sur un dataset de séparation complète construit en Phase 5 ; les tests Python-only continuent de passer sur une machine sans R installé.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNBCUrfDqrYGVDAgAU2QtIq6DIzW7UHAMBfHGt1V+fXEwAAXrseHCQGBEuErVgAAAAASUVORK5CYII=)  
**Phase 7 — AI Layer**  
**Dépendances :** uv add pydantic httpx + SDK du fournisseur LLM choisi  
**Fichiers à créer :** app/agent/llm_provider.py, app/agent/tools.py.  
**Tâches :**  
1. **app/agent/llm_provider.py** — le Protocol LLMProvider exact de la section 7, plus une implémentation concrète (ex. AnthropicProvider ou OpenAIProvider selon le fournisseur retenu) et une implémentation OllamaProvider (local, pour le hors-ligne).  
2. **app/agent/tools.py** — wrappers exposant au tool calling les fonctions déjà testées des phases précédentes, avec un schéma JSON de paramètres pour chacune :  
3. TOOL_SCHEMAS = {  
     "describe_variable": {...},  
     "run_chi_square": {...},  
     "run_linear_regression": {...},  
     "run_logistic_regression": {...},  
     # une entrée par fonction déjà validée dans app/statistics  
 }  
 def execute_tool(name: str, arguments: dict, data_dictionary: "DataDictionary",  
                   df: pd.DataFrame) -> "AnalysisResult":  
     """Route vers la bonne fonction de app.statistics selon `name`. C'est le  
     SEUL point d'entrée par lequel l'AI Layer déclenche un calcul."""  
   
**Garde-fou à faire valider par un test explicite :** aucune fonction de app/agent ne doit accepter un paramètre de type pd.DataFrame directement depuis un message utilisateur/LLM sans passer par execute_tool avec le dataset déjà chargé côté serveur — le LLM ne manipule que des noms de variables, jamais les données elles-mêmes.  
**Tests à écrire :** tests/test_agent/test_tools.py — validation de schéma Pydantic sur des payloads simulés, test que execute_tool route correctement vers chaque fonction.  
**Validation de fin de phase :** un appel LLM avec tool calling déclenche correctement run_linear_regression via execute_tool et le résultat structuré est correctement renvoyé au LLM pour explication.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OMQ2AABAAsSNBCkLfFR7wwIgHRiywEZJWQZeZ2ao9AAD+4lyruzq+ngAA8Nr1AOIEBeX8aGZPAAAAAElFTkSuQmCC)  
**Phase 8 — Analysis Planner**  
**Dépendances :** aucune nouvelle.  
**Fichiers à créer :** app/planner/models.py, app/planner/planner.py.  
**Tâches :**  
1. **app/planner/models.py** — AnalysisPlan exact de la section 7.  
2. **app/planner/planner.py** :  
3. def build_analysis_plan(question: str, data_dictionary: "DataDictionary",  
                          llm: "LLMProvider") -> AnalysisPlan:  
     """Utilise llm.complete() avec response_schema=AnalysisPlan. Le prompt  
     système doit lister explicitement les designs supportés (limite à 4-5 :  
     comparaison de groupes, association, régression linéaire simple,  
     régression logistique simple) et instruire le LLM de retourner  
     confidence='low' plutôt qu'un plan hasardeux si la question est ambiguë."""  
 def needs_clarification(plan: AnalysisPlan) -> bool:  
     """Retourne True si plan.confidence == 'low'."""  
   
**Tests à écrire :** tests/test_planner/test_planner.py — un jeu d'au moins 20 questions de recherche réelles (à constituer toi-même à partir de ton entourage académique) avec le plan attendu pour chacune, pour mesurer un taux de succès.  
**Validation de fin de phase :** taux de plans jugés corrects ≥ 75% sur le jeu de 20 questions ; les 25% restants déclenchent bien needs_clarification plutôt qu'un plan silencieusement faux.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OQQmAABRAsSd4NIGJjPWxpgGsYQVvImwJtszMXp0BAPAX91pt1fH1BACA164HhZwEOFrXVOsAAAAASUVORK5CYII=)  
**Phase 9 — Statistical Agent**  
**Dépendances :** aucune nouvelle.  
**Fichiers à créer :** app/agent/orchestrator.py.  
**Tâches :**  
1. **app/agent/orchestrator.py** :  
2. def propose_correction(diagnostics: list["DiagnosticFinding"],  
                         plan: "AnalysisPlan") -> dict | None:  
     """Si un diagnostic CERTAIN_ISSUE est présent (ex. séparation complète),  
     propose une correction concrète (ex. basculer vers run_firth_logistic)  
     sous forme de dict {"action": ..., "rationale": ..., "requires_confirmation": True}.  
     Ne retourne jamais une correction sans requires_confirmation=True (T14)."""  
 def execute_confirmed_action(action: dict, df: pd.DataFrame,  
                               data_dictionary: "DataDictionary") -> "AnalysisResult":  
     """N'exécute QUE si l'appelant a explicitement confirmé (paramètre  
     vérifié en amont) ; journalise l'action dans execution_log."""  
   
**Tests à écrire :** tests/test_agent/test_orchestrator.py — scénario complet séparation détectée → proposition Firth → confirmation simulée → ré-exécution, avec vérification que execute_confirmed_action refuse de s'exécuter sans confirmation explicite.  
**Validation de fin de phase :** démonstration end-to-end du scénario séparation → Firth, chaque étape loggée.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OQQmAABRAsSdYxKY/jMFMIZ7ECt5E2BJsmZmt2gMA4C+Otbqr8+sJAACvXQ85QgYXd/O+eQAAAABJRU5ErkJggg==)  
**Phase 10 — UI (Streamlit)**  
**Dépendances :** uv add streamlit plotly  
**Fichiers à créer :** app/ui/streamlit_app.py.  
**Tâches (à découper en plusieurs sous-tâches, une page/section à la fois) :**  
1. Page d'upload + affichage du dictionnaire de données (appelle build_data_dictionary, profile_dataset).  
2. Zone de chat (appelle build_analysis_plan, puis execute_tool).  
3. Zone de résultats (affiche AnalysisResult : tableau de coefficients, diagnostics avec code couleur par Severity, code généré, bouton d'export via generate_report).  
**Tests à écrire :** pas de tests unitaires exhaustifs sur l'UI Streamlit elle-même (hors périmètre raisonnable pour un solo dev) — un test manuel structuré : un testeur externe complète une analyse de bout en bout sans aide.  
**Validation de fin de phase :** uv run streamlit run app/ui/streamlit_app.py permet d'importer un CSV, poser une question, obtenir un résultat avec diagnostics et code, sans passer par la CLI.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhZscYahheJwqQgQU2QtIq6DIze3UGAMBf3Gu1VcfXEwAAXrseoqcEQXyAWBgAAAAASUVORK5CYII=)  
**Phase 11 — Reproductibilité v2**  
**Dépendances :** aucune nouvelle (json/sqlite3/pathlib stdlib).  
**Fichiers à créer :** app/reproducibility/project.py, app/reproducibility/history.py.  
**Tâches :**  
1. **app/reproducibility/project.py** :  
2. def save_analysis(project_dir: str, result: "AnalysisResult", plan: "AnalysisPlan | None") -> str:  
     """Persiste dataset (référence + hash), plan, code généré, résultat et  
     diagnostics dans project_dir/analyses/<id>/. Retourne l'id de l'analyse."""  
 def replay_analysis(project_dir: str, analysis_id: str) -> "AnalysisResult":  
     """Recharge le code généré archivé, le ré-exécute, et compare le nouveau  
     résultat à l'ancien à une tolérance numérique définie."""  
   
3. **app/reproducibility/history.py** — journal chronologique append-only des analyses d'un projet (JSON Lines).  
**Tests à écrire :** tests/test_reproducibility/test_replay.py — sauvegarder puis rejouer au moins 10 analyses de types différents (une par méthode déjà implémentée), vérifier l'égalité des résultats à la tolérance définie.  
**Validation de fin de phase :** replay_analysis réussit sur 10 analyses archivées différentes.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAM0lEQVR4nO3KsQ0AIRAEsUW6Qij1KvnevhMSYmKQ7GiCGd09k3wBAOAVf+2o4wYAwE1qAdYuAy151mgcAAAAAElFTkSuQmCC)  
**Phase 12 — Tests utilisateurs (pas de tâche de code dédiée)**  
Recrute 5 à 15 testeurs réels. Utilise un formulaire de retour structuré (Google Forms ou équivalent gratuit suffit — pas de développement nécessaire). Alimente un backlog GitHub Issues à partir des retours, priorisé avec les mêmes étiquettes MUST/SHOULD/COULD que le reste du projet.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OQQmAABRAsaeILbwZ9Fewo0Gs4E2ELcGWmTmqKwAA/uLeqr06v54AAPDa+gAthwNEfGhnhAAAAABJRU5ErkJggg==)  
**Phase 13 — Sécurité / Hardening**  
**Dépendances :** aucune nouvelle (resource, ast stdlib).  
**Fichiers à créer :** app/security/sandbox.py.  
**Tâches :**  
1. **app/security/sandbox.py** :  
2. def validate_generated_code(code: str) -> list[str]:  
     """Analyse l'AST du code généré (module ast) et retourne la liste des  
     violations trouvées : imports hors liste blanche (pandas, numpy, scipy,  
     statsmodels, matplotlib uniquement), usage de eval/exec/__import__/open  
     en écriture, appels réseau (socket, requests, urllib)."""  
 def run_sandboxed(code: str, timeout_seconds: int = 30,  
                    max_memory_mb: int = 512) -> dict:  
     """Exécute le code dans un subprocess isolé (pas le process principal),  
     avec limites imposées via le module resource (RLIMIT_CPU, RLIMIT_AS)  
     côté Linux, et un timeout dur via subprocess.run(timeout=...).  
     Retourne stdout/stderr/statut, ne lève jamais d'exception non gérée."""  
   
**Tests à écrire :** tests/test_security/test_sandbox.py — vérifier que validate_generated_code rejette un import interdit, qu'run_sandboxed stoppe bien une boucle infinie et une allocation mémoire excessive construites exprès pour le test.  
**Validation de fin de phase :** les deux scénarios malveillants de test (boucle infinie, sur-allocation mémoire) sont correctement stoppés par les limites en place.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AUBBAsUeCE4yeIiT9CRVMWGAjJK2CbjNzVGcAAPzF2qu7Wl9PAAB47XoA/vcF8exqpY4AAAAASUVORK5CYII=)  
**Phase 14 — Beta / Packaging (pas de tâche de code majeure)**  
Complète README.md (installation, usage, limitations connues), CHANGELOG.md, tag Git v1.0.0. Aucune nouvelle fonctionnalité ne doit être ajoutée pendant cette phase — uniquement documentation et correctifs.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAM0lEQVR4nO3OMQ0AIAwAwZIgBKm1gjSUlAUDTITkbvrxW2aOiJgBAMAvVj9RTzcAALhRG56UBSZiQftHAAAAAElFTkSuQmCC)  
**10. Annexe — Commandes utiles récapitulatives**  
# Environnement  
 uv sync                          # installe/synchronise les dépendances depuis uv.lock  
 uv add <package>                 # ajoute une dépendance  
 uv add --dev <package>           # ajoute une dépendance de développement  
 uv run <commande>                # exécute une commande dans l'environnement du projet  
   
 # Qualité  
 uv run pytest                    # lance tous les tests  
 uv run pytest tests/test_data/   # lance un sous-ensemble  
 uv run ruff check .              # lint  
 uv run ruff format .             # formatage automatique  
 uv run mypy app/                 # vérification de types  
   
 # Git  
 git checkout -b feature/xxx  
 git commit -m "feat(data): ajoute profile_dataset"  
 git tag v0.1.0 && git push --tags  
   
 # R (Phase 6+)  
 Rscript app/r_bridge/scripts/firth_logistic.R input.json output.json  
   
 # UI (Phase 10+)  
 uv run streamlit run app/ui/streamlit_app.py  
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OQQmAABRAsSfYxZo/jzlMYQLPJrCCNxG2BFtmZquOAAD4i3Ot7mr/egIAwGvXA4q7Bc870TqdAAAAAElFTkSuQmCC)  
**11. Annexe — Checklist de validation globale**  
- Phase 1 : pytest/ruff passent en CI, ADR-001/002 rédigées  
- Phase 2 : profiling correct sur 3 datasets réels dont un "sale"  
- Phase 3 : 5 méthodes validées contre R à une tolérance documentée  
- Phase 4 : rapport HTML + code reproductible générés pour chaque méthode  
- Phase 5 : corpus de cas limites déclenche les bons diagnostics, pas de faux positif sur données propres — **jalon MVP**  
- Phase 6 : Firth validé contre R, dégradation propre sans R installé  
- Phase 7 : tool calling route correctement vers les fonctions existantes, aucun DataFrame brut envoyé au LLM  
- Phase 8 : ≥75% de plans corrects sur 20 questions réelles  
- Phase 9 : scénario séparation→Firth démontré, confirmation obligatoire vérifiée par test  
- Phase 10 : analyse complète réalisable par un testeur externe sans aide  
- Phase 11 : rejeu réussi sur 10 analyses archivées  
- Phase 12 : ≥5 testeurs externes ont complété une analyse  
- Phase 13 : scénarios malveillants de test correctement stoppés  
- Phase 14 : v1.0.0 taggée, documentation complète  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OQQmAABRAsad4EEtY9QcxnUms4E2ELcGWmTmrKwAA/uLeqrU6vp4AAPDa/gDzXgM37EF77AAAAABJRU5ErkJggg==)  
*Fin du document. Contexte permanent à recoller à chaque session Qwen : sections 2, 6, 7.*  
