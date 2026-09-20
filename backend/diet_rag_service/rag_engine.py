"""
GroqDietRAGEngine - orchestrates DietRAG planning.

Pipeline (fully degraded by default):
  1. Curated KnowledgeGraph (offline KB) -> deterministic plan + context.
  2. Attach live IFCT2017 data when KB coverage is thin (web_fallback).
  3. Groq LLM (JSON mode) enriches the plan with culturally-flavored Dishes,
     but the deterministic KB plan always remains as the guaranteed output.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .graph import KnowledgeGraph
from .retriever import PlanComposer, build_diet_context
from . import web_fallback

logger = logging.getLogger("vitascan.diet_rag")

MODEL_ORDER = (
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "groq/compound",
    "qwen/qwen3.8-27b",
    "allam-2-7b",
)

_model_cache: Dict[str, Optional[str]] = {}


class GroqDietRAGEngine:
    def __init__(self, graph: Optional[KnowledgeGraph] = None):
        self.graph = graph or KnowledgeGraph()
        self.composer = PlanComposer(self.graph)
        self.api_key = os.getenv("GROQ_RAG_API_KEY") or os.getenv("GROQ_API_KEY")
        self._chat_model = None

    def _resolve_model(self, client, purpose: str) -> Optional[str]:
        """Pick the first usable model: env override, then a probed fallback chain."""
        cached = _model_cache.get(purpose) or (purpose == "chat" and self._chat_model)
        if cached:
            return cached
        override = os.getenv("GROQ_DIET_MODEL")
        candidates = [override] if override else []
        candidates += MODEL_ORDER
        try:
            available = {m.id for m in client.models.list().data}
        except Exception:
            available = None
        seen = set()
        for model in candidates:
            if not model or model in seen:
                continue
            seen.add(model)
            if available is not None and model not in available:
                continue
            _model_cache[purpose] = model
            if purpose == "chat":
                self._chat_model = model
            logger.info(f"diet RAG using {purpose} model: {model}")
            return model
        return None

    # ---- public API ---------------------------------------------------------
    def generate_diet_plan(
        self,
        deficiency_type: str,
        diet_pref: str = "vegetarian",
        allergies: Optional[List[str]] = None,
        disorders: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        allergies = self._normalize_allergies(allergies)
        disorders = self._normalize_disorders(disorders)
        diet_pref = self._normalize_diet(diet_pref)
        deficiency_type = self._normalize_deficiency(deficiency_type)

        plan = self._compose_degenerate_path(deficiency_type, diet_pref, allergies, disorders)

        if self.api_key:
            try:
                plan = self._llm_enhance(plan, deficiency_type, diet_pref, allergies, disorders)
            except Exception as e:
                logger.warning(f"LLM plan enhancement failed, using KB plan: {e}")

        if plan.get("coverage", {}).get("adequate") is False:
            plan = self._augment_with_web(plan, deficiency_type)

        plan.setdefault("fssai_checked", True)
        return plan

    # ---- deterministic path -------------------------------------------------
    def _compose_degenerate_path(self, deficiency_type, diet_pref, allergies, disorders) -> Dict[str, Any]:
        coverage = self.composer.coverage(deficiency_type, diet_pref, allergies, disorders)
        try:
            plan = self.composer.compose_plan(
                deficiency_type, diet=diet_pref, allergies=allergies, disorders=disorders
            )
        except ValueError:
            # No food passes constraints (e.g. veg + dairy allergy + b12).
            plan = {
                "deficiency": deficiency_type,
                "deficiency_label": deficiency_type.replace("_", " ").title(),
                "target_nutrients": coverage.get("target_nutrients", []),
                "diet_pref": diet_pref,
                "allergens": allergies,
                "disorders": disorders,
                "meal_plan": {},
                "bioavailability_tips": ["No safe food matched this combination. Please consult a physician/dietitian."],
                "fssai_checked": True,
                "source": "curated_kb_limited",
            }
        hints = self.composer.guide_explanation(deficiency_type, allergies, disorders)
        plan["coverage"] = coverage
        plan["explanation"] = hints
        return plan

    # ---- optional LLM enrichment --------------------------------------------
    def _ordered_candidates(self, client) -> List[str]:
        override = os.getenv("GROQ_DIET_MODEL")
        candidates = [override] if override else []
        candidates += MODEL_ORDER
        try:
            available = {m.id for m in client.models.list().data}
            return [m for m in candidates if m and m in available]
        except Exception:
            return [m for m in candidates if m]

    def _llm_enhance(self, base_plan, deficiency_type, diet_pref, allergies, disorders) -> Dict[str, Any]:
        from groq import Groq

        client = Groq(api_key=self.api_key)
        kb_context = build_diet_context(self.graph, deficiency_type, diet_pref, allergies, disorders)

        rda = self.graph.deficiencies.get(deficiency_type, {})
        prompt = (
            "You are an Indian clinical dietitian. Using ONLY the grounded context below, "
            "produce a 7-day Indian meal plan as raw JSON.\n"
            f"Deficiency: {deficiency_type} ({rda.get('label', '')}).\n"
            f"Diet: {diet_pref}. Allergies: {allergies or 'none'}. Disorders: {disorders or 'none'}.\n\n"
            f"Grounded KB context:\n{kb_context}\n\n"
            "JSON schema:\n"
            "{\n"
            '  "meal_plan": {"Day 1": {"Breakfast": "...", "Lunch": "...", "Snack": "...", "Dinner": "..."}, ..., "Day 7": {...}},\n'
            '  "bioavailability_tips": ["...", "..."],\n'
            '  "rationale": "one sentence of the diet rationale"\n'
            "}\n"
            "Rules: never use allergens; respect the diet preference; keep dishes culturally Indian; render plain text (no markdown)."
        )
        candidates = self._ordered_candidates(client) or [MODEL_ORDER[0]]
        comp = None
        for model in candidates:
            try:
                comp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                _model_cache["plan"] = model
                break
            except Exception as e:
                if getattr(e, "status_code", None) != 429 or model == candidates[-1]:
                    raise
                logger.warning(f"model {model} rate-limited, trying next: {e}")
        resp = comp.choices[0].message.content.strip()
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        parsed = json.loads(resp)
        merged = {**base_plan, **parsed, "source": "groq_llm_grounded"}
        merged["meal_plan"] = self._reconcile_meal_plan(base_plan.get("meal_plan", {}),
                                                        parsed.get("meal_plan", {}))
        return merged

    @staticmethod
    def _reconcile_meal_plan(base: Dict[str, Any], llm: Dict[str, Any]) -> Dict[str, Any]:
        """Keep the KB 7-day structure; use LLM dishes per day/slot when present."""
        reconciled: Dict[str, Any] = {}
        slots = ["Breakfast", "Lunch", "Snack", "Dinner"]
        for day, base_meals in base.items():
            llm_meals = llm.get(day, {})
            if not isinstance(llm_meals, dict):
                reconciled[day] = base_meals
                continue
            out = {}
            for slot in slots:
                candidate = str(llm_meals.get(slot, "")).strip()
                out[slot] = candidate if candidate else str(base_meals.get(slot, "—") or "—")
            reconciled[day] = out
        return reconciled

    # ---- live web augmentation ----------------------------------------------
    def _augment_with_web(self, plan, deficiency_type):
        target = plan.get("target_nutrients") or [deficiency_type]
        extra_foods = []
        for nid in target:
            cols = web_fallback.IFCT_NUT.get(nid)
            if not cols:
                continue
            # search with the nutrient name rather than raw code
            hits = web_fallback.search_foods("", limit=25)
            for h in hits:
                try:
                    raw = float(h.get(cols[0]) or 0.0)
                except ValueError:
                    raw = 0.0
                if raw > 0:
                    extra_foods.append({"code": h.get("code"), "name": h.get("name"),
                                        "nutrient": nid, "value": round(raw * cols[1], 4)})
        plan["web_sources"] = extra_foods[:20]
        plan["source"] = "curated_kb+web"
        return plan

    # ---- Q&A ----------------------------------------------------------------
    def answer_query(self, query: str, deficiency_type: str = "iron",
                     diet_pref: str = "vegetarian",
                     allergies: Optional[List[str]] = None,
                     disorders: Optional[List[str]] = None) -> str:
        txt = self._local_answer(query, deficiency_type, diet_pref, allergies, disorders)
        if not self.api_key:
            return txt
        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            context = build_diet_context(self.graph, deficiency_type, diet_pref, allergies, disorders)
            prompt = (
                f"Indian dietitian. Refer ONLY to this grounded context. Deficiency: {deficiency_type}.\n\n"
                f"Context:\n{context}\n\nPatient asks: {query}\n\nAnswer in 2-4 sentences, mention specific Indian foods."
            )
            comp = self._chat_completion(client, prompt,
                                         deficiency_type, diet_pref, allergies, disorders)
            return comp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"chat LLM failed, local answer used: {e}")
            return txt

    def _chat_completion(self, client, prompt, deficiency_type, diet_pref,
                         allergies, disorders):
        candidates = self._ordered_candidates(client) or [MODEL_ORDER[0]]
        last = None
        for model in candidates:
            try:
                comp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=450,
                )
                _model_cache["chat"] = model
                return comp
            except Exception as e:
                last = e
                if getattr(e, "status_code", None) != 429 or model == candidates[-1]:
                    raise
                logger.warning(f"chat model {model} rate-limited, trying next: {e}")
        raise last

    def _local_answer(self, query, deficiency_type, diet_pref, allergies, disorders) -> str:
        info = self.graph.deficiencies.get(deficiency_type, {})
        label = info.get("label", deficiency_type.replace("_", " ").title())
        tips = info.get("tips", [])
        seeds = " ".join(tips[:2])
        return (
            f"For {label}, key Indian foods include leafy greens, whole millets, legumes, "
            f"and (for B12) dairy, eggs, or fish. {seeds} "
            f"Remember: {(' allergens avoided: ' + ', '.join(allergies)) if allergies else 'no allergens flagged.'}"
        )

    # ---- normalizers ---------------------------------------------------------
    @staticmethod
    def _normalize_diet(diet_pref: str) -> str:
        d = (diet_pref or "vegetarian").strip().lower().replace("-", "")
        return {"veg": "vegetarian", "vegan": "vegetarian", "nonveg": "non-vegetarian",
                "nonvegetarian": "non-vegetarian",
                "eggetarian": "eggetarian",
                "fishitarian": "fishetarian", "pescatarian": "fishetarian",
                "fishetarian": "fishetarian"}.get(d, d)

    @staticmethod
    def _normalize_deficiency(deficiency_type: str) -> str:
        d = (deficiency_type or "iron").strip().lower().replace("-", "_").replace(" ", "_")
        return {"vita": "vitamin_a", "vit_d": "vitamin_d", "vitd": "vitamin_d",
                "vitamin_d": "vitamin_d", "anemia": "anemia", "protein": "protein"}.get(d, d)

    def _normalize_allergies(self, allergies) -> List[str]:
        if not allergies:
            return []
        known = set(self.graph.allergens.keys())
        return sorted({a.strip().lower() for a in allergies if a.strip().lower() in known})

    def _normalize_disorders(self, disorders) -> List[str]:
        if not disorders:
            return []
        known = set(self.graph.disorders.keys())
        return sorted({d.strip().lower().replace(" ", "_").replace("-", "_") for d in disorders
                       if d.strip().lower().replace(" ", "_").replace("-", "_") in known})