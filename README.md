# 📊 StatForge — IDE Statistique Intelligent

**StatForge** est un environnement de développement intégré (IDE) pour l'analyse statistique, alimenté par un agent LLM. Il permet à des non-spécialistes de poser des questions en langage naturel sur leurs données et d'obtenir des analyses statistiques rigoureuses, reproductibles et sécurisées.

## 🎯 Principe fondamental

> **L'IA PROPOSE → LE MOTEUR CALCULE → LE VALIDATEUR CONTRÔLE → L'UTILISATEUR DÉCIDE**

StatForge respecte une règle architecturale stricte : le LLM ne calcule **jamais** directement une statistique. Il propose un plan d'analyse, mais tous les calculs sont exécutés par des fonctions statistiques déterministes (scipy, statsmodels), validées par un système de diagnostics, et soumises à l'approbation explicite de l'utilisateur.

## ✨ Fonctionnalités

### Analyse statistique
- **6 méthodes** implémentées :
  - Test du χ² d'indépendance
  - Test t de Student
  - ANOVA à un facteur
  - Régression linéaire (OLS)
  - Régression logistique binaire
  - Régression logistique multinomiale
- Planification automatique via LLM (Groq)
- Détection automatique des types de variables (numérique / catégorielle)
- Diagnostics automatiques (normalité, homoscédasticité, multicollinéarité, etc.)
- Corrections proposées avec **confirmation humaine obligatoire**

### Reproductibilité (Phase 11)
- Sauvegarde complète des analyses au format JSON
- Rejeu à l'identique via `execute_tool`
- Historique persistant (`projects/history.jsonl`)
- Compteur de rejeux par projet
- Comparaison visuelle original vs rejeu

### Sécurité (Phase 13)
- Sandbox d'exécution isolé (subprocess)
- Validation AST du code généré (liste blanche d'imports)
- Limites de ressources : timeout CPU, mémoire
- Blocage des opérations dangereuses (réseau, système, `eval`, `exec`)

### Export
- Rapports HTML et Markdown
- Graphiques de coefficients (matplotlib)
- Téléchargement du code reproductible
- Exécution sandboxée du code dans l'interface

## 🏗️ Architecture
StatForge/
├── app/
│   ├── agent/          # LLM provider, tools registry, orchestrator
│   ├── data/           # Chargement, profiling, dictionnaire de données
│   ├── diagnostics/    # Modèles et calculateurs de diagnostics
│   ├── planner/        # Planification d'analyse via LLM
│   ├── reports/        # Génération de rapports et visualisations
│   ├── reproducibility/# Sauvegarde, rejeu, historique
│   ├── results/        # Modèles de résultats (AnalysisResult)
│   ├── security/       # Sandbox et validation de code
│   ├── statistics/     # Fonctions de calcul (tests, régressions)
│   └── ui/             # Interface Streamlit
├── projects/           # Analyses sauvegardées
├── reports/            # Rapports et graphiques générés
├── tests/              # Tests unitaires et d'intégration
├── pyproject.toml      # Configuration uv / dépendances
└── README.md


## 📋 Prérequis

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (gestionnaire de paquets et d'environnements)
- **Clé API Groq** (pour le LLM) — gratuite sur [console.groq.com](https://console.groq.com)
- Navigateur web moderne (Firefox, Chrome)

##  Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/<ton-username>/StatForge.git
cd StatForge

# 2. Créer l'environnement virtuel et installer les dépendances
uv sync

# 3. Configurer la clé API Groq
export GROQ_API_KEY="gsk_..."

# 4. Lancer l'application
uv run streamlit run app/ui/streamlit_app.py
