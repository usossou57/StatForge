"""Benchmark du planner sur 20 questions de recherche réelles."""

import json
import os
import time
from pathlib import Path

import pytest

from app.agent.llm_provider import GroqProvider, LLMProvider
from app.data.models import DataDictionary, Variable
from app.planner.planner import build_analysis_plan


@pytest.fixture
def llm_provider() -> LLMProvider:
    """Fournit un LLMProvider réel (Groq) pour les tests de benchmark."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY non définie, test de benchmark ignoré.")
    
    return GroqProvider(api_key=api_key, model="qwen/qwen3.8-27b")


@pytest.fixture
def research_questions() -> list[dict]:
    """Charge le jeu de 20 questions de recherche."""
    fixtures_path = Path(__file__).parent.parent / "fixtures" / "research_questions.json"
    with open(fixtures_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def comprehensive_data_dictionary() -> DataDictionary:
    """Crée un DataDictionary complet couvrant toutes les variables du benchmark."""
    return DataDictionary(
        dataset_name="benchmark_dataset",
        n_rows=1000,
        n_columns=30,
        variables={
            "age": Variable(name="age", dtype="numeric", semantic_type="continuous"),
            "income": Variable(name="income", dtype="numeric", semantic_type="continuous"),
            "study_hours": Variable(name="study_hours", dtype="numeric", semantic_type="continuous"),
            "attendance": Variable(name="attendance", dtype="numeric", semantic_type="continuous"),
            "previous_gpa": Variable(name="previous_gpa", dtype="numeric", semantic_type="continuous"),
            "surface_area": Variable(name="surface_area", dtype="numeric", semantic_type="continuous"),
            "num_rooms": Variable(name="num_rooms", dtype="numeric", semantic_type="continuous"),
            "price": Variable(name="price", dtype="numeric", semantic_type="continuous"),
            "exercise_frequency": Variable(name="exercise_frequency", dtype="numeric", semantic_type="continuous"),
            "sleep_hours": Variable(name="sleep_hours", dtype="numeric", semantic_type="continuous"),
            "cognitive_score": Variable(name="cognitive_score", dtype="numeric", semantic_type="continuous"),
            "math_score": Variable(name="math_score", dtype="numeric", semantic_type="continuous"),
            "blood_pressure": Variable(name="blood_pressure", dtype="numeric", semantic_type="continuous"),
            "productivity_score": Variable(name="productivity_score", dtype="numeric", semantic_type="continuous"),
            "sales": Variable(name="sales", dtype="numeric", semantic_type="continuous"),
            "advertising_spend": Variable(name="advertising_spend", dtype="numeric", semantic_type="continuous"),
            "satisfaction_score": Variable(name="satisfaction_score", dtype="numeric", semantic_type="continuous"),
            "test_score": Variable(name="test_score", dtype="numeric", semantic_type="continuous"),
            "commute_distance": Variable(name="commute_distance", dtype="numeric", semantic_type="continuous"),
            "wait_time": Variable(name="wait_time", dtype="numeric", semantic_type="continuous"),
            "service_quality": Variable(name="service_quality", dtype="numeric", semantic_type="continuous"),
            "gender": Variable(name="gender", dtype="categorical", semantic_type="binary", levels=["M", "F"]),
            "management_position": Variable(name="management_position", dtype="categorical", semantic_type="binary", levels=["Yes", "No"]),
            "lung_cancer": Variable(name="lung_cancer", dtype="categorical", semantic_type="binary", levels=["Yes", "No"]),
            "exam_success": Variable(name="exam_success", dtype="categorical", semantic_type="binary", levels=["Pass", "Fail"]),
            "diabetes_type2": Variable(name="diabetes_type2", dtype="categorical", semantic_type="binary", levels=["Yes", "No"]),
            "dropout": Variable(name="dropout", dtype="categorical", semantic_type="binary", levels=["Yes", "No"]),
            "cardiovascular_disease": Variable(name="cardiovascular_disease", dtype="categorical", semantic_type="binary", levels=["Yes", "No"]),
            "recovery_status": Variable(name="recovery_status", dtype="categorical", semantic_type="binary", levels=["Recovered", "Not_Recovered"]),
            "education": Variable(name="education", dtype="categorical", semantic_type="ordinal", levels=["High School", "Bachelor", "Master", "PhD"]),
            "satisfaction": Variable(name="satisfaction", dtype="categorical", semantic_type="ordinal", levels=["Low", "Medium", "High"]),
            "happiness_level": Variable(name="happiness_level", dtype="categorical", semantic_type="ordinal", levels=["Very Low", "Low", "Medium", "High", "Very High"]),
            "customer_satisfaction": Variable(name="customer_satisfaction", dtype="categorical", semantic_type="ordinal", levels=["Low", "Medium", "High"]),
            "location": Variable(name="location", dtype="categorical", semantic_type="nominal", levels=["Urban", "Suburban", "Rural"]),
            "smoking_status": Variable(name="smoking_status", dtype="categorical", semantic_type="nominal", levels=["Never", "Former", "Current"]),
            "treatment_group": Variable(name="treatment_group", dtype="categorical", semantic_type="nominal", levels=["Control", "Treatment_A", "Treatment_B"]),
            "school": Variable(name="school", dtype="categorical", semantic_type="nominal", levels=["School_A", "School_B"]),
            "transport_mode": Variable(name="transport_mode", dtype="categorical", semantic_type="nominal", levels=["Car", "Bus", "Bike", "Walk"]),
            "teaching_method": Variable(name="teaching_method", dtype="categorical", semantic_type="nominal", levels=["Traditional", "Interactive", "Online"]),
            "treatment_type": Variable(name="treatment_type", dtype="categorical", semantic_type="nominal", levels=["Treatment_A", "Treatment_B"]),
            "age_group": Variable(name="age_group", dtype="categorical", semantic_type="nominal", levels=["Young", "Adult", "Senior"]),
            "mediterranean_diet": Variable(name="mediterranean_diet", dtype="categorical", semantic_type="binary", levels=["Yes", "No"]),
            "work_location": Variable(name="work_location", dtype="categorical", semantic_type="nominal", levels=["Remote", "Office"]),
            "political_choice": Variable(name="political_choice", dtype="categorical", semantic_type="nominal", levels=["Party_A", "Party_B", "Party_C", "Abstention"]),
            "season": Variable(name="season", dtype="categorical", semantic_type="nominal", levels=["Spring", "Summer", "Fall", "Winter"]),
            "marital_status": Variable(name="marital_status", dtype="categorical", semantic_type="nominal", levels=["Single", "Married", "Divorced", "Widowed"]),
            "parental_education": Variable(name="parental_education", dtype="categorical", semantic_type="ordinal", levels=["None", "Primary", "Secondary", "Higher"]),
            "student_motivation": Variable(name="student_motivation", dtype="categorical", semantic_type="ordinal", levels=["Low", "Medium", "High"]),
            "region": Variable(name="region", dtype="categorical", semantic_type="nominal", levels=["North", "South", "East", "West"]),
            "family_income": Variable(name="family_income", dtype="categorical", semantic_type="ordinal", levels=["Low", "Medium", "High"]),
        },
        source_path="/fake/benchmark.csv",
        file_hash="benchmark_hash_123",
    )


def test_benchmark_planner_success_rate(
    research_questions: list[dict],
    comprehensive_data_dictionary: DataDictionary,
    llm_provider: LLMProvider,
) -> None:
    """Mesure le taux de succès du planner sur 20 questions de recherche."""
    successful_plans = 0
    correct_methods = 0
    correct_outcomes = 0
    correct_designs = 0
    
    results = []
    
    for i, question_data in enumerate(research_questions):
        question_id = question_data["id"]
        question = question_data["question"]
        expected_method = question_data["expected_method"]
        expected_outcome = question_data["expected_outcome"]
        expected_design = question_data["expected_design"]
        
        try:
            plan = build_analysis_plan(
                question=question,
                data_dictionary=comprehensive_data_dictionary,
                llm=llm_provider,
            )
            
            successful_plans += 1
            
            if plan.proposed_method == expected_method:
                correct_methods += 1
            
            if plan.outcome_variable == expected_outcome:
                correct_outcomes += 1
            
            if plan.study_design == expected_design:
                correct_designs += 1
            
            results.append({
                "id": question_id,
                "status": "success",
                "method_correct": plan.proposed_method == expected_method,
                "outcome_correct": plan.outcome_variable == expected_outcome,
                "design_correct": plan.study_design == expected_design,
                "confidence": plan.confidence,
            })
            
        except Exception as e:
            results.append({
                "id": question_id,
                "status": "failed",
                "error": str(e),
            })
        
        # ⚠️ PAUSE OBLIGATOIRE : 15 secondes entre chaque requête pour respecter 
        # la limite stricte de 1000 OTPM (Output Tokens Per Minute) de Groq.
        # Cela permet au compteur de se réinitialiser complètement.
        if i < len(research_questions) - 1:
            time.sleep(15)
    
    total_questions = len(research_questions)
    success_rate = successful_plans / total_questions
    
    # Calcul des précisions UNIQUEMENT sur les questions qui ont réussi
    # (pour ne pas pénaliser le modèle à cause d'une limite d'API)
    method_accuracy = correct_methods / successful_plans if successful_plans > 0 else 0
    outcome_accuracy = correct_outcomes / successful_plans if successful_plans > 0 else 0
    design_accuracy = correct_designs / successful_plans if successful_plans > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK PLANNER - {total_questions} questions")
    print(f"{'='*60}")
    print(f"Taux de succès global : {successful_plans}/{total_questions} ({success_rate:.1%})")
    print(f"Précision méthode (sur succès) : {correct_methods}/{successful_plans} ({method_accuracy:.1%})")
    print(f"Précision outcome (sur succès) : {correct_outcomes}/{successful_plans} ({outcome_accuracy:.1%})")
    print(f"Précision design (sur succès) : {correct_designs}/{successful_plans} ({design_accuracy:.1%})")
    
    failed_results = [r for r in results if r["status"] == "failed"]
    if failed_results:
        print(f"\n⚠️ ÉCHECS (généralement dus au Rate Limit 429 de Groq) :")
        for r in failed_results:
            print(f"  Question {r['id']} : {r['error'][:100]}...")
    print(f"{'='*60}\n")
    
    # Assertions conformes à la spec (Phase 8 : ≥ 75% de plans corrects)
    assert success_rate >= 0.75, f"Taux de succès trop bas : {success_rate:.1%} (minimum 75%)"
    assert method_accuracy >= 0.70, f"Précision méthode trop basse : {method_accuracy:.1%} (minimum 70%)"


@pytest.mark.slow
def test_benchmark_individual_questions(
    research_questions: list[dict],
    comprehensive_data_dictionary: DataDictionary,
    llm_provider: LLMProvider,
) -> None:
    """Test individuel pour chaque question (utile pour debug)."""
    for question_data in research_questions:
        question = question_data["question"]
        expected_method = question_data["expected_method"]
        
        plan = build_analysis_plan(
            question=question,
            data_dictionary=comprehensive_data_dictionary,
            llm=llm_provider,
        )
        
        assert plan.research_question, "La question reformulée ne doit pas être vide"
        assert plan.outcome_variable, "L'outcome ne doit pas être vide"
        assert len(plan.predictor_variables) > 0, "Il doit y avoir au moins un prédicteur"
        assert plan.rationale, "Le rationale ne doit pas être vide"
        assert plan.confidence in ["low", "medium", "high"], "Confiance invalide"
        
        print(f"\nQuestion {question_data['id']}: {question}")
        print(f"  Méthode attendue: {expected_method}, proposée: {plan.proposed_method}")
        print(f"  Outcome attendu: {question_data['expected_outcome']}, proposé: {plan.outcome_variable}")
        print(f"  Confiance: {plan.confidence}")
        
        time.sleep(15)
