"""Application Streamlit principale de StatForge.

Interface utilisateur pour l'IDE Statistique, respectant les contraintes :
- T8 : aucun DataFrame brut n'est envoyé au LLM
- T14 : confirmation humaine obligatoire pour les corrections
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from app.data.profiling import build_data_dictionary, profile_dataset


def _generate_print_suffix(method: str) -> str:
    """Génère un bloc print() adapté à la méthode statistique.

    Args:
        method: Nom de la méthode (ex: 'chi_square', 'linear_regression').

    Returns:
        Code Python à ajouter à la fin du script pour afficher les résultats.
    """
    suffixes = {
        "chi_square": (
            "print('=== Résultats du test du chi-carré ===')\n"
            "print(f'Statistique chi2 : {chi2:.4f}')\n"
            "print(f'p-value : {p:.4e}')\n"
            "print(f'Degrés de liberté : {dof}')\n"
            "print('\\nTable de contingence :')\n"
            "print(contingency)\n"
        ),
        "t_test": (
            "print('=== Résultats du test t ===')\n"
            "print(f'Statistique t : {t_stat:.4f}')\n"
            "print(f'p-value : {p_value:.4e}')\n"
            "print(f'Degrés de liberté : {dof}')\n"
        ),
        "anova": (
            "print('=== Résultats de l\\'ANOVA ===')\n"
            "print(f'Statistique F : {f_stat:.4f}')\n"
            "print(f'p-value : {p_value:.4e}')\n"
            "print(f'Degrés de liberté (inter) : {df_between}')\n"
            "print(f'Degrés de liberté (intra) : {df_within}')\n"
        ),
        "linear_regression": (
            "print('=== Résultats de la régression linéaire ===')\n"
            "print(model.summary())\n"
        ),
        "logistic_regression": (
            "print('=== Résultats de la régression logistique ===')\n"
            "print(model.summary())\n"
        ),
        "multinomial_logistic_regression": (
            "print('=== Résultats de la régression logistique multinomiale ===')\n"
            "print(model.summary())\n"
        ),
    }
    return suffixes.get(method, "")


# ──────────────────────────────────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="StatForge — IDE Statistique",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────
# Initialisation de la session
# ──────────────────────────────────────────────────────────────────────

if "file_uploaded" not in st.session_state:
    st.session_state["file_uploaded"] = False

if "df" not in st.session_state:
    st.session_state["df"] = None

if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = None

if "data_dictionary" not in st.session_state:
    st.session_state["data_dictionary"] = None

if "profile" not in st.session_state:
    st.session_state["profile"] = None

if "temp_file_path" not in st.session_state:
    st.session_state["temp_file_path"] = None

if "file_extension" not in st.session_state:
    st.session_state["file_extension"] = ".csv"

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "current_plan" not in st.session_state:
    st.session_state["current_plan"] = None

if "current_result" not in st.session_state:
    st.session_state["current_result"] = None

if "current_project_id" not in st.session_state:
    st.session_state["current_project_id"] = None

if "just_updated_project_id" not in st.session_state:
    st.session_state["just_updated_project_id"] = None

if "just_updated_count" not in st.session_state:
    st.session_state["just_updated_count"] = 0

# Navigation : clés stables (SANS émojis) pour éviter les mismatch Unicode
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "upload"

if "navigate_to" not in st.session_state:
    st.session_state["navigate_to"] = None


# ──────────────────────────────────────────────────────────────────────
# Sidebar — Navigation robuste
# ──────────────────────────────────────────────────────────────────────

st.sidebar.title("📊 StatForge")
st.sidebar.markdown("IDE Statistique Intelligent")
st.sidebar.markdown("---")

# Mapping clés stables ↔ labels avec émojis
PAGES_KEYS = ["upload", "analyse", "rapport"]
PAGES_LABELS = ["📁 Upload & Dictionnaire", "🔍 Analyse", "📝 Rapport"]

# Navigation programmatique (consommée une seule fois)
if st.session_state["navigate_to"] is not None:
    st.session_state["current_page"] = st.session_state["navigate_to"]
    st.session_state["navigate_to"] = None

# Vérification défensive : si la clé n'est pas valide, on reset
if st.session_state["current_page"] not in PAGES_KEYS:
    st.session_state["current_page"] = "upload"

# Index de la page courante
current_index = PAGES_KEYS.index(st.session_state["current_page"])

# Key dynamique pour forcer la recréation du widget quand la page change
radio_key = f"nav_radio_{st.session_state['current_page']}"

# Radio de navigation
selected_label = st.sidebar.radio(
    "Navigation",
    PAGES_LABELS,
    index=current_index,
    key=radio_key,
)

# Synchronisation : label sélectionné → clé stable
st.session_state["current_page"] = PAGES_KEYS[PAGES_LABELS.index(selected_label)]

# Bouton de réinitialisation
if st.sidebar.button("🗑️ Réinitialiser la session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state["current_page"] = "upload"
    st.session_state["navigate_to"] = None
    st.rerun()


# ─────────────────────────────────────────────────────────────────────
# Page 1 : Upload & Dictionnaire de données
# ──────────────────────────────────────────────────────────────────────

if st.session_state["current_page"] == "upload":
    st.title("📁 Upload de fichier et dictionnaire de données")
    st.markdown(
        "Importez votre fichier de données (CSV, Excel, SPSS) pour générer "
        "automatiquement un dictionnaire de données structuré."
    )
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Choisissez un fichier",
        type=["csv", "xlsx", "xls", "sav"],
        help="Formats supportés : CSV, Excel (XLSX/XLS), SPSS (SAV)",
    )

    if uploaded_file is not None or st.session_state["file_uploaded"]:
        if uploaded_file is not None:
            file_extension = Path(uploaded_file.name).suffix.lower()

            try:
                if file_extension == ".csv":
                    df = pd.read_csv(uploaded_file)
                elif file_extension in (".xlsx", ".xls"):
                    df = pd.read_excel(uploaded_file)
                elif file_extension == ".sav":
                    try:
                        import pyreadstat

                        df, _ = pyreadstat.read_sav(uploaded_file)
                    except ImportError:
                        st.error(
                            "Le support des fichiers SPSS (.sav) nécessite le package "
                            "`pyreadstat`. Installez-le avec : `uv add pyreadstat`"
                        )
                        st.stop()
                else:
                    st.error(f"Format non supporté : {file_extension}")
                    st.stop()

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_extension, prefix="statforge_"
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                st.session_state["df"] = df
                st.session_state["dataset_name"] = uploaded_file.name
                st.session_state["file_uploaded"] = True
                st.session_state["temp_file_path"] = tmp_path
                st.session_state["file_extension"] = file_extension

                st.session_state["data_dictionary"] = None
                st.session_state["profile"] = None
                st.session_state["chat_history"] = []
                st.session_state["current_plan"] = None
                st.session_state["current_result"] = None

                st.success(
                    f"✅ Fichier chargé avec succès : {len(df)} lignes, {len(df.columns)} colonnes"
                )

            except Exception as e:
                st.error(f"❌ Erreur lors de la lecture du fichier : {e}")
                st.exception(e)
                st.stop()

        df = st.session_state["df"]
        dataset_name = st.session_state["dataset_name"]

        if df is None:
            st.error("❌ Données non disponibles. Veuillez re-uploader votre fichier.")
            st.stop()

        st.subheader("👁️ Aperçu des données")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown(f"*Affichage des 10 premières lignes sur {len(df)}.*")

        st.subheader(" Dictionnaire de données")
        st.markdown(
            "Le dictionnaire de données est généré automatiquement à partir "
            "de la structure du fichier. Il sera utilisé par l'agent LLM "
            "pour proposer des analyses (sans jamais envoyer les données brutes)."
        )

        if st.session_state["data_dictionary"] is None:
            temp_path = st.session_state["temp_file_path"]
            try:
                data_dictionary = build_data_dictionary(
                    df=df,
                    dataset_name=dataset_name,
                    source_path=temp_path,
                )
                profile = profile_dataset(df)
                st.session_state["data_dictionary"] = data_dictionary
                st.session_state["profile"] = profile
            except Exception as e:
                st.error(f"❌ Erreur lors de la génération du dictionnaire : {e}")
                st.exception(e)
                st.stop()

        data_dictionary = st.session_state["data_dictionary"]
        profile = st.session_state["profile"]

        st.markdown(f"**Dataset :** {data_dictionary.dataset_name}")
        st.markdown(
            f"**Taille :** {data_dictionary.n_rows} lignes × {data_dictionary.n_columns} colonnes"
        )
        st.markdown(f"**Hash :** `{data_dictionary.file_hash[:12]}...`")
        st.markdown("---")

        st.markdown("### Variables détectées")
        variables_data = []
        for var_name, var_info in data_dictionary.variables.items():
            if var_name in df.columns:
                missing_count = int(df[var_name].isna().sum())
            else:
                missing_count = 0

            variables_data.append(
                {
                    "Nom": var_name,
                    "Type": var_info.dtype,
                    "Type sémantique": var_info.semantic_type or "N/A",
                    "Modalités": ", ".join(var_info.levels[:5])
                    if var_info.levels
                    else "N/A",
                    "Valeurs manquantes": missing_count,
                }
            )

        variables_df = pd.DataFrame(variables_data)
        st.dataframe(variables_df, use_container_width=True)

        st.subheader("📊 Statistiques descriptives")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Variables numériques**")
            numeric_vars = [
                var_name
                for var_name, var_info in data_dictionary.variables.items()
                if var_info.dtype == "numeric" and var_name in df.columns
            ]
            if numeric_vars:
                st.dataframe(df[numeric_vars].describe(), use_container_width=True)
            else:
                st.info("Aucune variable numérique détectée.")

        with col2:
            st.markdown("**Variables catégorielles**")
            categorical_vars = [
                var_name
                for var_name, var_info in data_dictionary.variables.items()
                if var_info.dtype == "categorical" and var_name in df.columns
            ]
            if categorical_vars:
                for var in categorical_vars[:5]:
                    st.markdown(f"**{var}** : {df[var].nunique()} modalités uniques")
                    st.text(df[var].value_counts().head(3).to_string())
                    st.markdown("---")
            else:
                st.info("Aucune variable catégorielle détectée.")

    else:
        st.info("👆 Uploadez un fichier pour commencer.")
        st.markdown("---")
        st.subheader("📋 Structure attendue")
        st.markdown("""
        Votre fichier doit contenir :
        - **Variables numériques** : âge, revenu, score, etc.
        - **Variables catégorielles** : genre, groupe, traitement, etc.
        - **Variable cible (outcome)** : la variable à expliquer/prédire
        """)


# ──────────────────────────────────────────────────────────────────────
# Page 2 : Analyse (T10.2)
# ──────────────────────────────────────────────────────────────────────

elif st.session_state["current_page"] == "analyse":
    st.title("🔍 Analyse statistique")

    if not st.session_state.get("file_uploaded", False):
        st.warning(
            "⚠️ Veuillez d'abord uploader un fichier dans la page 'Upload & Dictionnaire'."
        )
        st.stop()

    df = st.session_state.get("df")
    data_dictionary = st.session_state.get("data_dictionary")

    if df is None or data_dictionary is None:
        st.error(
            "❌ Données ou dictionnaire manquant. Veuillez re-uploader votre fichier."
        )
        st.stop()

    st.markdown(f"**Dataset :** {st.session_state.get('dataset_name', 'Inconnu')}")
    st.markdown(f"**Variables disponibles :** {len(data_dictionary.variables)}")
    st.markdown("---")

    st.subheader("1. Question de recherche")
    st.markdown("Posez votre question en langage naturel.")

    research_question = st.text_area(
        "Votre question :",
        height=100,
        placeholder="Ex: Quel est l'effet de l'âge et du niveau d'éducation sur le revenu annuel ?",
        key="research_question_input",
    )

    st.subheader("2. Plan d'analyse proposé")

    if st.button(
        "🤖 Générer un plan d'analyse", type="primary", key="generate_plan_btn"
    ):
        if not research_question:
            st.warning("Veuillez d'abord poser une question.")
        else:
            with st.spinner("Génération du plan d'analyse via le LLM..."):
                try:
                    from app.agent.llm_provider import GroqProvider
                    from app.planner.planner import build_analysis_plan

                    api_key = os.environ.get("GROQ_API_KEY")
                    if not api_key:
                        st.error(
                            "❌ Clé API Groq non configurée. Définissez la variable GROQ_API_KEY."
                        )
                        st.stop()

                    llm = GroqProvider(api_key=api_key)
                    plan = build_analysis_plan(
                        question=research_question,
                        data_dictionary=data_dictionary,
                        llm=llm,
                    )

                    st.session_state["current_plan"] = plan
                    st.session_state["research_question"] = research_question
                    st.session_state["chat_history"].append(
                        {"role": "user", "content": research_question}
                    )
                    st.success("✅ Plan d'analyse généré avec succès !")

                except Exception as e:
                    st.error(f"❌ Erreur lors de la génération du plan : {e}")
                    st.exception(e)

    if st.session_state["current_plan"] is not None:
        plan = st.session_state["current_plan"]
        from app.planner.planner import needs_clarification

        if needs_clarification(plan):
            st.warning(
                "⚠️ Le plan a une confiance faible. Une clarification est nécessaire."
            )
            st.markdown("**Question reformulée par le LLM :**")
            st.info(plan.research_question)
            st.markdown("**Raison de la faible confiance :**")
            st.info(plan.rationale)

            clarification = st.text_area(
                "Veuillez clarifier votre question :",
                height=100,
                placeholder="Ex: Je veux étudier l'effet de l'âge sur le revenu, en contrôlant pour le genre.",
                key="clarification_input",
            )

            if st.button(
                "🔄 Régénérer le plan avec la clarification", key="regenerate_plan_btn"
            ):
                if clarification:
                    st.session_state["research_question"] = clarification
                    st.rerun()
                else:
                    st.warning("Veuillez fournir une clarification.")
        else:
            st.markdown("### Plan proposé")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Question reformulée :** {plan.research_question}")
                st.markdown(f"**Méthode proposée :** `{plan.proposed_method}`")
                st.markdown(f"**Design d'étude :** {plan.study_design}")
                st.markdown(f"**Confiance :** {plan.confidence}")

            with col2:
                st.markdown(f"**Variable cible (outcome) :** `{plan.outcome_variable}`")
                st.markdown(
                    f"**Prédicteurs ({len(plan.predictor_variables)}) :** {', '.join(f'`{p}`' for p in plan.predictor_variables)}"
                )
                if plan.reference_category:
                    st.markdown(
                        f"**Catégorie de référence :** `{plan.reference_category}`"
                    )

            st.markdown("### Justification")
            st.info(plan.rationale)

            if plan.assumptions_to_check:
                st.markdown("### Hypothèses théoriques de la méthode")
                st.markdown(
                    "*Ces hypothèses sont listées par le LLM à titre indicatif.*"
                )
                for assumption in plan.assumptions_to_check:
                    st.markdown(f"- {assumption}")

            st.markdown("---")
            st.subheader("3. Exécution de l'analyse")

            if st.button(
                "▶️ Exécuter l'analyse", type="primary", key="execute_analysis_btn"
            ):
                with st.spinner("Exécution de l'analyse statistique..."):
                    try:
                        from app.agent.tools import execute_tool

                        method_name = plan.proposed_method
                        if not method_name.startswith(
                            "run_"
                        ) and not method_name.startswith("check_"):
                            method_name = f"run_{method_name}"

                        if method_name == "run_chi_square":
                            arguments = {
                                "var1": plan.outcome_variable,
                                "var2": plan.predictor_variables[0]
                                if plan.predictor_variables
                                else None,
                            }
                        elif method_name in ("run_t_test", "run_anova"):
                            arguments = {
                                "outcome": plan.outcome_variable,
                                "group": plan.predictor_variables[0]
                                if plan.predictor_variables
                                else None,
                            }
                        else:
                            arguments = {
                                "outcome": plan.outcome_variable,
                                "predictors": plan.predictor_variables,
                            }
                            if plan.reference_category:
                                arguments["reference_category"] = (
                                    plan.reference_category
                                )

                        result = execute_tool(
                            name=method_name, arguments=arguments, df=df
                        )

                        st.session_state["current_result"] = result
                        st.session_state["chat_history"].append(
                            {
                                "role": "assistant",
                                "content": f"Analyse exécutée : {plan.proposed_method}",
                            }
                        )
                        st.success("✅ Analyse exécutée avec succès !")

                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'exécution : {e}")
                        st.exception(e)

            if st.session_state["current_result"] is not None:
                result = st.session_state["current_result"]

                st.markdown("---")
                st.subheader("4. Résultats de l'analyse")

                st.markdown("### Métriques du modèle")
                if result.model_metrics:
                    col1, col2, col3 = st.columns(3)
                    metrics_list = list(result.model_metrics.items())
                    for i, (metric, value) in enumerate(metrics_list[:3]):
                        with [col1, col2, col3][i]:
                            st.metric(
                                label=metric.replace("_", " ").title(),
                                value=f"{value:.4f}"
                                if isinstance(value, float)
                                else value,
                            )
                else:
                    st.info("Aucune métrique disponible.")

                st.markdown("### Coefficients estimés")
                if result.estimates:
                    coefficients_data = []
                    for var_name in result.estimates.keys():
                        estimate = result.estimates[var_name]
                        se = result.standard_errors.get(var_name, 0.0)
                        p_value = result.p_values.get(var_name, 1.0)
                        ci = result.confidence_intervals.get(var_name, (0.0, 0.0))

                        coefficients_data.append(
                            {
                                "Variable": var_name,
                                "Estimation": f"{estimate:.4f}",
                                "Erreur Std.": f"{se:.4f}",
                                "IC 95%": f"[{ci[0]:.4f}, {ci[1]:.4f}]",
                                "p-value": f"{p_value:.4e}",
                                "Significatif": "✅" if p_value < 0.05 else "❌",
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(coefficients_data), use_container_width=True
                    )
                else:
                    st.info("Aucun coefficient disponible.")

                if result.diagnostics:
                    st.markdown("### Diagnostics")
                    for diag in result.diagnostics:
                        severity_emoji = {
                            "certain_issue": "🔴",
                            "probable_issue": "🟠",
                            "warning": "🟡",
                            "information": "🔵",
                        }.get(diag.severity.value, "")
                        st.markdown(
                            f"{severity_emoji} **[{diag.severity.value}]** {diag.code}: {diag.message}"
                        )

                if result.warnings:
                    st.markdown("### Avertissements")
                    for warning in result.warnings:
                        st.warning(warning)

                st.markdown("### Code reproductible")
                st.code(result.generated_code, language="python")

                from app.agent.orchestrator import propose_correction

                correction_proposal = propose_correction(
                    diagnostics=result.diagnostics, plan=plan
                )

                if correction_proposal:
                    st.markdown("---")
                    st.subheader("⚠️ Correction proposée")
                    st.warning(
                        f"**Action :** {correction_proposal['action']}\n\n"
                        f"**Raison :** {correction_proposal['rationale']}\n\n"
                        f"**Déclenchée par :** {correction_proposal['triggered_by']} "
                        f"(sévérité: {correction_proposal['triggered_severity']})"
                    )

                    confirmation = st.checkbox(
                        "Je confirme que je souhaite appliquer cette correction",
                        key="correction_confirmation",
                    )

                    if st.button(
                        "✅ Appliquer la correction", disabled=not confirmation
                    ):
                        with st.spinner("Application de la correction..."):
                            try:
                                from app.agent.orchestrator import (
                                    execute_confirmed_action,
                                )

                                corrected_result = execute_confirmed_action(
                                    action=correction_proposal,
                                    df=df,
                                    data_dictionary=data_dictionary,
                                    confirmed=True,
                                )
                                st.session_state["current_result"] = corrected_result
                                st.success("✅ Correction appliquée avec succès !")
                                st.rerun()
                            except Exception as e:
                                st.error(
                                    f"❌ Erreur lors de l'application de la correction : {e}"
                                )
                                st.exception(e)

                st.markdown("---")
                # Redirection programmatique robuste vers la page Rapport
                if st.button(
                    "📝 Générer un rapport complet", key="generate_report_btn"
                ):
                    st.session_state["navigate_to"] = "rapport"
                    st.rerun()

    if st.session_state["chat_history"]:
        st.markdown("---")
        st.subheader("💬 Historique de la conversation")
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f"**Vous :** {msg['content']}")
            elif msg["role"] == "assistant":
                st.markdown(f"**Assistant :** {msg['content']}")

        if st.button("🗑️ Effacer l'historique", key="clear_history_btn"):
            st.session_state["chat_history"] = []
            st.session_state["current_plan"] = None
            st.session_state["current_result"] = None
            st.rerun()


# ──────────────────────────────────────────────────────────────────────
# Page 3 : Rapport (T10.3 + T11.1 + T11.2 + T13 + T14)
# ──────────────────────────────────────────────────────────────────────

elif st.session_state["current_page"] == "rapport":
    st.title("📝 Génération de rapport")

    if not st.session_state.get("file_uploaded", False):
        st.warning(
            "⚠️ Veuillez d'abord uploader un fichier dans la page 'Upload & Dictionnaire'."
        )
        st.stop()

    if st.session_state.get("current_result") is None:
        st.warning("⚠️ Veuillez d'abord exécuter une analyse dans la page 'Analyse'.")
        st.markdown("---")
        st.info(
            "Cette page génère un rapport professionnel à partir des résultats "
            "d'une analyse statistique. Commencez par poser une question et "
            "exécuter l'analyse dans la page ' Analyse'."
        )
        st.stop()

    # Récupération défensive des variables au début de la page Rapport
    result = st.session_state["current_result"]
    dataset_name = st.session_state.get("dataset_name", "Inconnu")
    df = st.session_state.get("df")
    temp_file_path = st.session_state.get("temp_file_path")
    file_extension = st.session_state.get("file_extension", ".csv")

    st.markdown(f"**Dataset :** {dataset_name}")
    st.markdown(f"**Méthode :** {result.method}")
    st.markdown("---")

    st.subheader("1. Format du rapport")
    report_format = st.radio(
        "Choisissez le format de sortie :",
        ["HTML", "Markdown"],
        horizontal=True,
        key="report_format",
    )

    st.subheader("2. Rapport généré")
    try:
        from app.reports.generator import generate_report

        report_content = generate_report(
            result=result, output_format=report_format.lower()
        )

        if report_format == "HTML":
            st.components.v1.html(report_content, height=800, scrolling=True)
        else:
            st.markdown(report_content)

        st.markdown("---")
        st.subheader("3. Téléchargement")

        if report_format == "HTML":
            st.download_button(
                label="📥 Télécharger le rapport HTML",
                data=report_content,
                file_name=f"rapport_{result.method}_{dataset_name.replace('.', '_')}.html",
                mime="text/html",
            )
        else:
            st.download_button(
                label="📥 Télécharger le rapport Markdown",
                data=report_content,
                file_name=f"rapport_{result.method}_{dataset_name.replace('.', '_')}.md",
                mime="text/markdown",
            )
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération du rapport : {e}")
        st.exception(e)

    st.markdown("---")
    st.subheader("4. Visualisations statistiques")
    try:
        from app.reports.visualizer import generate_coefficient_plot

        plot_path = generate_coefficient_plot(
            result=result,
            output_dir="reports/plots",
            filename=f"coef_plot_{result.method}.png",
        )

        if Path(plot_path).exists():
            st.image(
                plot_path,
                caption=f"Visualisation — {result.method}",
                use_container_width=True,
            )
            with open(plot_path, "rb") as f:
                st.download_button(
                    label="📥 Télécharger le graphique (PNG)",
                    data=f.read(),
                    file_name=f"visualisation_{result.method}.png",
                    mime="image/png",
                )
        else:
            st.info("Aucun graphique généré.")
    except Exception as e:
        st.warning(f"️ Impossible de générer le graphique : {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Étape 5 : Code reproductible + exécution sandboxée (T13 + T14)
    # ──────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("5. Code reproductible et exécution sécurisée")
    st.markdown(
        "Ce code Python peut être exécuté indépendamment pour reproduire l'analyse. "
        "Cliquez sur le bouton ci-dessous pour l'exécuter dans un environnement sécurisé (sandbox)."
    )

    st.code(result.generated_code, language="python")

    st.download_button(
        label="📥 Télécharger le script Python",
        data=result.generated_code,
        file_name=f"analyse_{result.method}.py",
        mime="text/plain",
    )

    # Exécution sécurisée via le sandbox (Phase 13/14)
    st.markdown("---")
    st.subheader("▶️ Exécution dans le sandbox de sécurité")

    if st.button(
        "▶️ Exécuter le code reproductible", type="primary", key="run_sandbox_btn"
    ):
        try:
            from app.security.sandbox import run_sandboxed

            sandbox_code = result.generated_code

            if temp_file_path and Path(temp_file_path).exists():
                # 1. Préfixer avec le chargement du fichier
                if file_extension == ".csv":
                    load_code = (
                        f'import pandas as pd\ndf = pd.read_csv(r"{temp_file_path}")\n'
                    )
                elif file_extension in (".xlsx", ".xls"):
                    load_code = f'import pandas as pd\ndf = pd.read_excel(r"{temp_file_path}")\n'
                elif file_extension == ".sav":
                    load_code = f'import pandas as pd\nimport pyreadstat\ndf, _ = pyreadstat.read_sav(r"{temp_file_path}")\n'
                else:
                    load_code = (
                        f'import pandas as pd\ndf = pd.read_csv(r"{temp_file_path}")\n'
                    )

                sandbox_code = load_code + sandbox_code

                # 2. Ajouter automatiquement un print() à la fin selon la méthode
                print_suffix = _generate_print_suffix(result.method)
                if print_suffix:
                    sandbox_code += "\n" + print_suffix

            with st.spinner("Exécution sécurisée en cours (timeout 15s)..."):
                sandbox_result = run_sandboxed(
                    code=sandbox_code, timeout_seconds=15, max_memory_mb=512
                )

            st.markdown("### Résultat de l'exécution")

            if sandbox_result["success"]:
                st.success("✅ Code exécuté avec succès !")
                if sandbox_result["stdout"].strip():
                    st.markdown("**Sortie standard (stdout) :**")
                    st.code(sandbox_result["stdout"], language="bash")
                else:
                    st.info("Le code s'est exécuté sans produire de sortie texte.")
            else:
                st.error("❌ L'exécution a échoué.")

            if sandbox_result["stderr"].strip():
                st.markdown("**Sortie d'erreur / Avertissements (stderr) :**")
                st.warning(sandbox_result["stderr"])

        except ValueError as e:
            st.error(f"️ **Code bloqué par le sandbox de sécurité :**\n{e}")
            st.markdown(
                "*Le code généré contient des opérations interdites (réseau, système, eval, etc.).*"
            )
        except Exception as e:
            st.error(f"❌ Erreur inattendue lors de l'exécution : {e}")
            st.exception(e)

    # ──────────────────────────────────────────────────────────────────────
    # Étape 6 : Métadonnées et traçabilité
    # ──────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("6. Métadonnées et traçabilité")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Informations techniques**")
        st.markdown(f"- **Méthode :** {result.method}")
        st.markdown(f"- **Taille d'échantillon :** {result.sample_size}")
        st.markdown(f"- **Variables :** {len(result.variables)}")

    with col2:
        st.markdown("**Versions logicielles**")
        if result.software_versions:
            for software, version in result.software_versions.items():
                st.markdown(f"- **{software}** : {version}")
        else:
            st.info("Aucune information de version disponible.")

    # ──────────────────────────────────────────────────────────────────────
    # Étape 7 : Sauvegarde du projet (T11.1)
    # ──────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("7. Sauvegarde du projet")
    st.markdown(
        "Sauvegardez cette analyse pour pouvoir la rejouer à l'identique plus tard."
    )

    if st.button("💾 Sauvegarder l'analyse", key="save_analysis_btn"):
        try:
            from app.reproducibility.history import append_to_history
            from app.reproducibility.project import (
                build_project_from_analysis,
                save_analysis,
            )

            # Récupération défensive locale
            loc_dataset_name = st.session_state.get("dataset_name", "Inconnu")
            loc_df = st.session_state.get("df")

            dataset_rows = (
                len(loc_df)
                if loc_df is not None
                else (result.sample_size if hasattr(result, "sample_size") else 0)
            )
            dataset_columns = len(loc_df.columns) if loc_df is not None else 0

            project = build_project_from_analysis(
                result=result,
                research_question=st.session_state.get("research_question", "Inconnue"),
                dataset_name=loc_dataset_name,
                dataset_rows=dataset_rows,
                dataset_columns=dataset_columns,
                plan=st.session_state.get("current_plan"),
            )
            project_path = save_analysis(project, output_dir="projects")
            st.session_state["current_project_id"] = project.project_id
            st.success(f"✅ Projet sauvegardé : `{project_path}`")

            append_to_history(
                project_dir="projects",
                project_id=project.project_id,
                research_question=project.research_question,
                method=project.method,
                dataset_name=project.dataset_name,
                dataset_hash=project.dataset_hash,
                project_file=project_path,
                status="success",
            )

            with open(project_path, "r", encoding="utf-8") as f:
                project_json = f.read()

            st.download_button(
                label="📥 Télécharger le projet (JSON)",
                data=project_json,
                file_name=f"{project.project_id}.json",
                mime="application/json",
            )

        except Exception as e:
            st.error(f"❌ Erreur lors de la sauvegarde : {e}")
            st.exception(e)

    # ──────────────────────────────────────────────────────────────────────
    # Étape 8 : Rejeu de l'analyse (T11.1) — Version déterministe
    # ──────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("8. Rejeu de l'analyse")
    st.markdown(
        "Rejouez cette analyse à l'identique via `execute_tool` (T9). "
        "Les résultats doivent être strictement identiques à l'exécution originale."
    )

    if st.button("🔄 Rejouer l'analyse", key="replay_analysis_btn"):
        try:
            import hashlib
            import json
            from pathlib import Path

            from app.reproducibility.history import append_to_history, get_history
            from app.reproducibility.project import (
                build_project_from_analysis,
                replay_analysis,
                save_analysis,
            )

            # Récupération défensive
            loc_dataset_name = st.session_state.get("dataset_name", "Inconnu")
            loc_df = st.session_state.get("df")

            dataset_rows = len(loc_df) if loc_df is not None else result.sample_size
            dataset_columns = len(loc_df.columns) if loc_df is not None else 0

            # ── ID DÉTERMINISTE (Méthode + Dataset) ────────────────────
            stable_input = (result.method + loc_dataset_name).encode("utf-8")
            stable_hash = hashlib.md5(stable_input).hexdigest()[:12]
            project_id = f"statforge_{stable_hash}"

            # Construction et sauvegarde
            project = build_project_from_analysis(
                result=result,
                research_question=st.session_state.get("research_question", "Inconnue"),
                dataset_name=loc_dataset_name,
                dataset_rows=dataset_rows,
                dataset_columns=dataset_columns,
                plan=st.session_state.get("current_plan"),
                project_id=project_id,
            )
            project_path = save_analysis(project, output_dir="projects")

            # Vérification historique
            entries = get_history("projects")
            if not any(e["project_id"] == project_id for e in entries):
                append_to_history(
                    project_dir="projects",
                    project_id=project_id,
                    research_question=project.research_question,
                    method=project.method,
                    dataset_name=project.dataset_name,
                    dataset_hash=project.dataset_hash,
                    project_file=project_path,
                    status="success",
                )

            # Rejeu
            with st.spinner("Rejeu de l'analyse..."):
                replayed_result = replay_analysis(project, loc_df)

            st.success("✅ Analyse rejouée avec succès !")

            # ── MISE À JOUR INLINE DU COMPTEUR ─────────────────────────
            history_file = Path("projects") / "history.jsonl"
            if history_file.exists():
                lines = history_file.read_text(encoding="utf-8").splitlines()
                new_lines = []
                updated = False
                final_count = 0

                for line in lines:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if entry.get("project_id") == project_id:
                        entry["replay_count"] = entry.get("replay_count", 0) + 1
                        entry["status"] = "replayed"
                        updated = True
                        final_count = entry["replay_count"]
                    new_lines.append(json.dumps(entry, ensure_ascii=False))

                if updated:
                    history_file.write_text(
                        "\n".join(new_lines) + "\n", encoding="utf-8"
                    )
                    st.success(
                        f"✅ Rejeu réussi ! Compteur mis à jour : {final_count} rejeu(s)."
                    )

                    # Stockage en session pour synchronisation avec l'Étape 9
                    st.session_state["just_updated_project_id"] = project_id
                    st.session_state["just_updated_count"] = final_count

                else:
                    st.warning(
                        f"⚠️ Le project_id {project_id} n'a pas été trouvé dans history.jsonl"
                    )
            else:
                st.error(
                    "❌ Le fichier history.jsonl n'existe pas dans le dossier projects/"
                )

            # Comparaison
            st.markdown("### Comparaison original vs rejeu")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original**")
                for var_name, estimate in result.estimates.items():
                    st.markdown(f"- `{var_name}`: {estimate:.6f}")
            with col2:
                st.markdown("**Rejeu**")
                for var_name, estimate in replayed_result.estimates.items():
                    st.markdown(f"- `{var_name}`: {estimate:.6f}")

            estimates_match = all(
                abs(result.estimates.get(k, 0) - replayed_result.estimates.get(k, 0))
                < 1e-10
                for k in set(
                    list(result.estimates.keys())
                    + list(replayed_result.estimates.keys())
                )
            )

            if estimates_match:
                st.success(
                    "✅ Les résultats sont **identiques** (à 1e-10 près). Reproductibilité confirmée !"
                )
            else:
                st.warning("⚠️ Les résultats diffèrent légèrement.")

        except Exception as e:
            st.error(f"❌ Erreur lors du rejeu : {e}")
            st.exception(e)

    # ──────────────────────────────────────────────────────────────────────
    # Étape 9 : Historique des analyses (T11.2)
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("9. Historique des analyses")

    # Bouton pour rafraîchir l'affichage (utile après rechargement de page)
    if st.button(
        "🔄 Rafraîchir l'affichage de l'historique",
        key="refresh_history_btn",
        type="secondary",
    ):
        st.rerun()

    try:
        from app.reproducibility.history import get_history, get_history_stats

        stats = get_history_stats("projects")
        entries = get_history("projects")

        if stats["total_analyses"] > 0:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total analyses", stats["total_analyses"])
            with col2:
                st.metric("Méthodes différentes", len(stats["by_method"]))
            with col3:
                # CORRECTION : calculer la somme des replay_count au lieu du count des entrées "replayed"
                total_replays = sum(entry.get("replay_count", 0) for entry in entries)
                st.metric("Rejeux effectués", total_replays)

            if stats["by_method"]:
                st.markdown("**Répartition par méthode :**")
                for method, count in sorted(stats["by_method"].items()):
                    st.markdown(f"- `{method}` : {count}")

            st.markdown("---")
            st.markdown("**Analyses récentes :**")

            for entry in entries[:10]:
                # Synchronisation avec le compteur mis à jour à l'Étape 8
                display_count = entry.get("replay_count", 0)
                if (
                    st.session_state.get("just_updated_project_id")
                    == entry["project_id"]
                ):
                    display_count = st.session_state.get(
                        "just_updated_count", display_count
                    )

                with st.expander(
                    f"{entry['project_id']} — {entry['method']} ({entry['status']})",
                    expanded=False,
                ):
                    st.markdown(f"**Question :** {entry['research_question']}")
                    st.markdown(f"**Dataset :** {entry['dataset_name']}")
                    st.markdown(f"**Hash :** `{entry.get('dataset_hash', 'N/A')}`")
                    st.markdown(f"**Timestamp :** {entry['timestamp']}")
                    st.markdown(f"**Rejeux :** {display_count}")

                    if st.button(
                        "📂 Charger cette analyse", key=f"load_{entry['project_id']}"
                    ):
                        try:
                            from app.reproducibility.project import load_project

                            project = load_project(entry["project_file"])
                            st.session_state["current_result"] = project
                            st.session_state["research_question"] = (
                                project.research_question
                            )
                            st.success(f"Analyse {entry['project_id']} chargée !")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erreur lors du chargement : {e}")
        else:
            st.info("Aucune analyse sauvegardée pour le moment.")

    except Exception as e:
        st.warning(f"⚠️ Impossible de charger l'historique : {e}")
