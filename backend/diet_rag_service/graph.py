"""
KnowledgeGraph - lightweight in-memory graph over the curated diet KB.

Node types:
  - nutrient   (fe, folsum, vitb12, ...)
  - food       (curated food, keyed by code + name)
  - allergen   (dairy, egg, gluten, ...)
  - disorder   (lactose_intolerance, celiac, ...)
  - deficiency (iron, anemia, ...)

Edges:
  food ->(contains)-> nutrient            (with per-100g value + RDA on nutrient)
  allergen ->(in)-> food codes
  disorder ->(maps_to)-> allergens / tags / nutrient limits
  deficiency ->(targets)-> nutrients

No external graph library is required - plain dict adjacency maps.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("vitascan.diet_kb")


@dataclass
class FoodNode:
    code: str
    name: str
    group: str
    diets: List[str]
    allergens: List[str]
    slots: List[str]
    portion_g: int
    tags: List[str]
    dishes: List[str]
    nutrients: Dict[str, float]

    def nutrient_for_serving(self, nutrient: str) -> float:
        """Amount of `nutrient` in one typical portion (raw-gram basis)."""
        return self.nutrients.get(nutrient, 0.0) * self.portion_g / 100.0

    def compatible_with(self, diet_pref: str) -> bool:
        """True if this food is allowed for a diet preference string."""
        pref = (diet_pref or "non-vegetarian").strip().lower()
        # normalize eggetarian / lacto variants
        if pref in ("all", "any"):
            return True
        if pref == "nonveg":
            pref = "non-vegetarian"
        if pref == "veg":
            pref = "vegetarian"
        if pref in self.diets:
            return True
        # vegetarian-compatible foods are acceptable to eggetarians/fishetarians too
        if "vegetarian" in self.diets and pref in ("eggetarian", "fishetarian"):
            return True
        return False


@dataclass
class NutrientDef:
    key: str
    name: str
    unit: str
    rda: float
    max: bool = False


class KnowledgeGraph:
    def __init__(self, kb_dir: Optional[Path] = None):
        kb_dir = kb_dir or Path(__file__).parent / "kb"
        self.kb_dir = kb_dir
        self.foods: Dict[str, FoodNode] = {}
        self.nutrients: Dict[str, NutrientDef] = {}
        self.allergens: Dict[str, Dict[str, Any]] = {}
        self.disorders: Dict[str, Dict[str, Any]] = {}
        self.deficiencies: Dict[str, Dict[str, Any]] = {}

        # adjacency maps
        self.nutrient_to_foods: Dict[str, Set[str]] = {}
        self.allergen_to_foods: Dict[str, Set[str]] = {}
        self.def_to_nutrients: Dict[str, List[str]] = {}
        self.load()

    # ---- loading -----------------------------------------------------------
    def _load_json(self, name: str) -> Dict[str, Any]:
        p = self.kb_dir / name
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def load(self) -> None:
        foods_data = self._load_json("foods.json")
        nutrient_data = self._load_json("nutrients.json")
        allergen_data = self._load_json("allergies.json")
        disorder_data = self._load_json("disorders.json")
        def_data = self._load_json("deficiencies.json")

        for key, meta in nutrient_data.get("nutrients", {}).items():
            self.nutrients[key] = NutrientDef(key=key, **meta)
            self.nutrient_to_foods.setdefault(key, set())

        for code, node in self._build_foods(foods_data).items():
            self.foods[code] = node
            for nid in node.nutrients:
                self.nutrient_to_foods.setdefault(nid, set()).add(code)

        for akey, meta in allergen_data.get("allergies", {}).items():
            self.allergens[akey] = meta
            for code in meta.get("foods", []):
                self.allergen_to_foods.setdefault(akey, set()).add(code)

        self.disorders = disorder_data.get("disorders", disorder_data)
        self.deficiencies = def_data.get("deficiencies", def_data)
        for def_key, meta in self.deficiencies.items():
            self.def_to_nutrients[def_key] = meta.get("target_nutrients", [])

    def _build_foods(self, foods_data: Dict[str, Any]) -> Dict[str, FoodNode]:
        foods = foods_data.get("foods", [])
        return {
            f["code"]: FoodNode(
                code=f["code"],
                name=f["name"],
                group=f.get("group", ""),
                diets=f.get("diets", []),
                allergens=f.get("allergens", []),
                slots=f.get("slots", []),
                portion_g=f.get("portion_g", 100),
                tags=f.get("tags", []),
                dishes=f.get("dishes", []),
                nutrients={k: float(v) for k, v in f.get("nutrients_per_100g", {}).items()},
            )
            for f in foods
        }

    # ---- queries -----------------------------------------------------------
    def food(self, code: str) -> Optional[FoodNode]:
        return self.foods.get(code)

    def nutrients_for_definition(self, def_key: str) -> List[str]:
        return self.def_to_nutrients.get(def_key, [])

    def foods_containing(self, nutrient: str) -> List[FoodNode]:
        return [self.foods[c] for c in self.nutrient_to_foods.get(nutrient, []) if c in self.foods]

    def foods_for_allergen(self, allergen: str) -> List[FoodNode]:
        return [self.foods[c] for c in self.allergen_to_foods.get(allergen, []) if c in self.foods]

    def top_foods(self, nutrient: str, limit: int = 10, diet: str = "non-vegetarian",
                  excluded_allergens: Optional[Set[str]] = None,
                  preferred_tags: Optional[Set[str]] = None,
                  avoided_tags: Optional[Set[str]] = None) -> List[FoodNode]:
        """Foods ranked by per-serving contribution to `nutrient` RDA."""
        excluded = excluded_allergens or set()
        prefer = preferred_tags or set()
        avoid = avoided_tags or set()
        rda = self.nutrients[nutrient].rda if nutrient in self.nutrients else 1.0

        scored = []
        for f in self.foods_containing(nutrient):
            if not f.compatible_with(diet):
                continue
            if excluded & set(f.allergens):
                continue
            if avoid & set(f.tags):
                continue
            contrib = f.nutrient_for_serving(nutrient) / rda
            bonus = sum(1 for t in prefer if t in f.tags)
            scored.append((contrib * (1 + bonus), f))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:limit]]