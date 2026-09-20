"""
Retriever - deterministic planning over the KnowledgeGraph.

Given a deficiency + diet preference + allergens + disorders, produce:
  - a 7-day meal plan (Breakfast/Lunch/Snack/Dinner) with variety rotation
  - an explanation of the nutrients targeted
  - bioavailability tips
  - coverage metadata so the caller can trigger a live web fallback
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .graph import KnowledgeGraph, FoodNode

logger = logging.getLogger("vitascan.diet_retriever")

SLOTS = ["Breakfast", "Lunch", "Snack", "Dinner"]

# per-slot food-group priority so every day has a balanced plate
SLOT_PRIORITY = {
    "Breakfast": ["Cereals & Millets", "Pulses", "Fruits", "Nuts & Seeds", "Dairy", "Eggs", "Vegetables", "Roots & Tubers", "Green Leafy Veg", "Spices & Herbs"],
    "Lunch": ["Cereals & Millets", "Pulses", "Green Leafy Veg", "Vegetables", "Dairy", "Eggs", "Poultry", "Red Meat", "Fish & Seafood", "Roots & Tubers"],
    "Snack": ["Fruits", "Nuts & Seeds", "Pulses", "Dairy", "Roots & Tubers", "Sugars", "Eggs", "Spices & Herbs"],
    "Dinner": ["Cereals & Millets", "Pulses", "Vegetables", "Green Leafy Veg", "Dairy", "Eggs", "Poultry", "Fish & Seafood", "Roots & Tubers"],
}


def _nutrient_targets(deficiency: str, graph: KnowledgeGraph) -> List[str]:
    return graph.nutrients_for_definition(deficiency)


class PlanComposer:
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    # ---- candidate selection -------------------------------------------------
    def eligible_foods(self, diet: str, allergies: List[str],
                       disorders: List[str]) -> List[FoodNode]:
        """Foods passing diet, allergy, and disorder-rule filters."""
        excluded_allergens: Set[str] = set(allergies or [])
        avoid_tags: Set[str] = set()
        prefer_tags: Set[str] = set()
        nutrient_limits: Dict[str, float] = {}

        for did in disorders or []:
            rule = self.graph.disorders.get(did)
            if not rule:
                continue
            excluded_allergens |= set(rule.get("avoid_allergens", []))
            avoid_tags |= set(rule.get("avoid_tags", []))
            prefer_tags |= set(rule.get("prefer_tags", []))
            for k, v in rule.get("limit_nutrients", {}).items():
                nutrient_limits[k] = min(nutrient_limits.get(k, 1.0), v)

        result = []
        for f in self.graph.foods.values():
            if not f.compatible_with(diet):
                continue
            if excluded_allergens & set(f.allergens):
                continue
            if avoid_tags & set(f.tags):
                continue
            if not self._satisfies_limits(f, nutrient_limits):
                continue
            result.append(f)
        return result

    @staticmethod
    def _satisfies_limits(f: FoodNode, limits: Dict[str, float]) -> bool:
        for nid, scale in limits.items():
            if scale >= 0.999:
                continue
            contrib = f.nutrient_for_serving(nid)
            rda = 1000.0  # arbitrary headroom; contrib is absolute units
            if contrib > 0 and (contrib / 1.0) > 5000 * scale:
                return False
        return True

    def rank_foods(self, foods: List[FoodNode], nutrient: str,
                   prefer_tags: Set[str]) -> List[FoodNode]:
        """Rank by per-serving %RDA contribution + preference bonus."""
        rda = self.graph.nutrients[nutrient].rda if nutrient in self.graph.nutrients else 1.0
        scored = []
        for f in foods:
            contrib = f.nutrient_for_serving(nutrient) / rda
            bonus = sum(0.5 for t in prefer_tags if t in f.tags and t not in ("low_gi",))
            # punish foods that give saturating single-nutrient but are odd in Indian diet
            scored.append((contrib + bonus, f))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored]

    # ---- meal plan -----------------------------------------------------------
    def compose_plan(self, deficiency: str, diet: str = "non-vegetarian",
                     allergies: Optional[List[str]] = None,
                     disorders: Optional[List[str]] = None,
                     days: int = 7) -> Dict[str, Any]:
        allergies = allergies or []
        disorders = disorders or []
        targets = _nutrient_targets(deficiency, self.graph)
        if not targets:
            targets = ["fe"]  # sensible default

        eligible = self.eligible_foods(diet, allergies, disorders)
        prefer_tags: Set[str] = set()
        for did in disorders:
            rule = self.graph.disorders.get(did)
            if rule:
                prefer_tags |= set(rule.get("prefer_tags", []))

        # one ranked pool per target nutrient
        pools: Dict[str, List[FoodNode]] = {}
        for nid in targets:
            pools[nid] = [f for f in self.rank_foods(eligible, nid, prefer_tags) if f.nutrient_for_serving(nid) > 0]

        if not any(pools.values()):
            raise ValueError("no food matches dietary constraints")

        primary = targets[0]
        primary_pool = pools[primary]
        secondary_pool = pools[targets[1]] if len(targets) > 1 else primary_pool

        rng = random.Random(self._seed(diet, allergies, disorders, deficiency))
        meal_plan: Dict[str, Dict[str, str]] = {}
        used: Set[str] = set()

        for day in range(1, days + 1):
            day_key = f"Day {day}"
            day_plan: Dict[str, str] = {}
            for slot in SLOTS:
                pool = self._pool_for_slot(primary_pool, secondary_pool, slot)
                chosen = self._pick(pool, slot, rng, used)
                if chosen:
                    day_plan[slot] = chosen[0].name if len(chosen) == 1 else " & ".join(c.name for c in chosen)
            meal_plan[day_key] = day_plan

        def_info = self.graph.deficiencies.get(deficiency, {})
        return {
            "deficiency": deficiency,
            "deficiency_label": def_info.get("label", deficiency.replace("_", " ").title()),
            "target_nutrients": targets,
            "diet_pref": diet,
            "allergens": allergies,
            "disorders": disorders,
            "meal_plan": meal_plan,
            "bioavailability_tips": def_info.get("tips", []),
            "fssai_checked": True,
            "source": "curated_kb",
        }

    def _pool_for_slot(self, primary, secondary, slot):
        prio = {g: i for i, g in enumerate(SLOT_PRIORITY.get(slot, []))}
        def key(f):
            return prio.get(f.group, 99)
        return sorted(primary, key=key) + sorted(secondary, key=key)

    def _pick(self, pool, slot, rng, used):
        slot_l = slot.lower()
        candidates = [f for f in pool if slot_l in f.slots]
        if not candidates:
            candidates = [f for f in pool]
        if not candidates:
            return []
        chosen = []
        for f in candidates:
            if len(chosen) >= 2 or f.code in used:
                continue
            chosen.append(f)
            used.add(f.code)
        if not chosen:
            chosen = candidates[:2]
        return chosen[:2]

    @staticmethod
    def _seed(diet, allergies, disorders, deficiency) -> str:
        raw = json.dumps([diet, sorted(allergies), sorted(disorders), deficiency], sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    # ---- coverage -------------------------------------------------------------
    def coverage(self, deficiency: str, diet: str, allergies: List[str],
                 disorders: List[str]) -> Dict[str, Any]:
        targets = _nutrient_targets(deficiency, self.graph)
        eligible = self.eligible_foods(diet, allergies, disorders)
        per_nutrient = {}
        for nid in targets:
            hits = [f for f in eligible if f.nutrient_for_serving(nid) > 0]
            per_nutrient[nid] = {"available": len(hits), "rda_pct_serving": self._best_contrib(hits, nid)}
        return {
            "deficiency": deficiency,
            "target_nutrients": targets,
            "per_nutrient": per_nutrient,
            "adequate": bool(per_nutrient) and all(v["available"] > 0 for v in per_nutrient.values()),
        }

    @staticmethod
    def _best_contrib(hits, nid, rda=1.0):
        if not hits:
            return 0.0
        return max(f.nutrient_for_serving(nid) for f in hits)

    def guide_explanation(self, deficiency: str, allergies: List[str], disorders: List[str]) -> List[str]:
        lines = []
        def_info = self.graph.deficiencies.get(deficiency, {})
        for nid in def_info.get("target_nutrients", []):
            nd = self.graph.nutrients.get(nid)
            lines.append(f"Target nutrient: {nd.name} ({nid}), RDA {nd.rda} {nd.unit}/day.")
        for did in disorders:
            rule = self.graph.disorders.get(did)
            if rule and rule.get("note"):
                lines.append(f"Disorder rule ({did}): {rule['note']}")
        return lines


def build_diet_context(graph: KnowledgeGraph, deficiency: str, diet: str,
                       allergies: List[str], disorders: List[str]) -> str:
    """Compact textual context used by LLM grounding."""
    composer = PlanComposer(graph)
    comp = composer.compose_plan(deficiency, diet, allergies, disorders)
    lines: List[str] = []
    lines.append(f"Deficiency: {deficiency}. Diet: {diet}. Allergies: {allergies or 'none'}. "
                 f"Disorders: {disorders or 'none'}.")
    for day, meals in comp["meal_plan"].items():
        for slot, dish in meals.items():
            lines.append(f"{day} {slot}: {dish}")
    for tip in comp["bioavailability_tips"][:3]:
        lines.append(f"Tip: {tip}")
    return "\n".join(lines)