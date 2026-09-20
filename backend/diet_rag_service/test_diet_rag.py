import os
import sys
import json
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.diet_rag_service.graph import KnowledgeGraph
from backend.diet_rag_service.retriever import PlanComposer
from backend.diet_rag_service.rag_engine import GroqDietRAGEngine
from backend.diet_rag_service import web_fallback


def make_engine():
    return GroqDietRAGEngine(graph=KnowledgeGraph())


def test_plan_always_produces_seven_days():
    eng = make_engine()
    for deficiency in ("iron", "anemia", "b12", "folate", "vitamin_d", "calcium", "zinc"):
        plan = eng.generate_diet_plan(deficiency, "vegetarian")
        assert len(plan["meal_plan"]) == 7, deficiency
        for day, meals in plan["meal_plan"].items():
            assert set(meals) == {"Breakfast", "Lunch", "Snack", "Dinner"}, (deficiency, day)
        assert plan["fssai_checked"] is True


def test_allergens_are_respected():
    eng = make_engine()
    plan = eng.generate_diet_plan("iron", "vegetarian", allergies=["dairy", "soy"])
    text = json.dumps(plan["meal_plan"]).lower()
    for banned in ("paneer", "milk", "soya"):
        assert banned not in text, banned


def test_disorder_rules_are_applied():
    eng = make_engine()
    plan = eng.generate_diet_plan("iron", "vegetarian", disorders=["celiac"])
    text = json.dumps(plan["meal_plan"]).lower()
    assert "atta" not in text  # whole wheat excluded for gluten


def test_limited_coverage_flags_web_augmentation():
    eng = make_engine()
    plan = eng.generate_diet_plan("b12", "vegetarian", allergies=["dairy"])
    assert plan["coverage"]["adequate"] is False
    assert plan["source"] in ("curated_kb+web", "curated_kb_limited")


def test_unknown_allergy_and_disorder_are_ignored_safely():
    eng = make_engine()
    plan = eng.generate_diet_plan("iron", "vegetarian",
                                  allergies=["not_a_real_allergen"],
                                  disorders=["not_a_real_disorder"])
    assert len(plan["meal_plan"]) == 7


def test_llm_failure_degrades_to_kb_plan():
    eng = make_engine()
    plan = eng.generate_diet_plan("iron", "vegetarian")  # sanity: KB works without key
    assert plan["source"] == "curated_kb"
    with patch("groq.Groq", side_effect=RuntimeError("api down")):
        result = eng.generate_diet_plan("iron", "vegetarian")
    assert result["source"] == "curated_kb"
    assert len(result["meal_plan"]) == 7


def test_diet_normalization_maps_variants():
    eng = make_engine()
    assert eng._normalize_diet("veg") == "vegetarian"
    assert eng._normalize_diet("NonVeg") == "non-vegetarian"
    assert eng._normalize_diet("non-vegetarian") == "non-vegetarian"
    assert eng._normalize_diet("Fishitarian") == "fishetarian"
    assert eng._normalize_deficiency("Vit D") == "vitamin_d"
    # regressions
    assert eng._normalize_deficiency("iron") == "iron"
    assert eng._normalize_diet("nonveg") == "non-vegetarian"


def test_reconcile_meal_plan_keeps_7_days_and_fills_gaps():
    from backend.diet_rag_service.rag_engine import GroqDietRAGEngine
    base = {f"Day {i}": {"Breakfast": "b", "Lunch": "l", "Snack": "s", "Dinner": "d"} for i in range(1, 8)}
    malformed = {"Day 1": {"Breakfast": "Kala chana"}, "Day 2": "oops"}
    out = GroqDietRAGEngine._reconcile_meal_plan(base, malformed)
    assert len(out) == 7
    assert out["Day 1"]["Lunch"] == "l"      # gap filled from KB
    assert out["Day 1"]["Breakfast"] == "Kala chana"  # LLM value kept
    assert out["Day 3"]["Dinner"] == "d"     # missing LLM day fully from KB


def test_web_fallback_degrades_to_empty_on_failure():
    with patch("backend.diet_rag_service.web_fallback._rows", return_value=[]):
        assert web_fallback.search_foods("rice") == []
        assert web_fallback.find_nutrients_by_code("A001") is None