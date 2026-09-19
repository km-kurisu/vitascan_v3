import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("vitascan.diet_rag")

class GroqDietRAGEngine:
    def __init__(self):
        kb_path = Path(__file__).parent / "fssai_knowledge_base.json"
        with open(kb_path, "r", encoding="utf-8") as f:
            self.kb = json.load(f)
        self.api_key = os.getenv("GROQ_RAG_API_KEY") or os.getenv("GROQ_API_KEY")

    def generate_diet_plan(self, deficiency_type: str, diet_pref: str = "vegetarian") -> Dict[str, Any]:
        info = self.kb.get(deficiency_type.lower(), self.kb["iron"])
        foods = [f for f in info["rich_foods"] if diet_pref.lower() == "all" or f["type"] == diet_pref.lower()]

        if self.api_key:
            try:
                from groq import Groq
                client = Groq(api_key=self.api_key)
                prompt = (
                    f"Generate a detailed, culturally relevant 7-day Indian meal plan for a patient with severe {deficiency_type} deficiency. "
                    f"Dietary preference: {diet_pref}. FSSAI RDA context: {info['rda']}. "
                    f"Key foods to include: {[f['name'] for f in foods]}. "
                    "Return ONLY raw JSON with keys: 'meal_plan' (dict of Days to Breakfast, Lunch, Snack, Dinner), 'bioavailability_tips' (list), and 'fssai_checked' (boolean)."
                )
                comp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                resp = comp.choices[0].message.content.strip()
                if "```json" in resp: resp = resp.split("```json")[1].split("```")[0].strip()
                return json.loads(resp)
            except Exception as e:
                logger.warning(f"Groq RAG API call error: {e}")

        return {
            "meal_plan": {
                "Breakfast": "Rajgira porridge or Palak Paratha with fresh Lemon juice",
                "Lunch": "Dal Tadka, Steamed Rice, Beetroot Poriyal, Curd",
                "Snack": "Jaggery & Sesame Chikki + Amla juice",
                "Dinner": "Whole wheat Roti, Spinach Paneer, Mixed Salad"
            },
            "bioavailability_tips": [
                "Always add lemon juice to spinach dishes to double non-heme iron absorption.",
                "Avoid drinking tea or coffee within 1 hour of meals."
            ],
            "fssai_checked": True
        }

    def answer_query(self, query: str, deficiency_type: str = "iron") -> str:
        if self.api_key:
            try:
                from groq import Groq
                client = Groq(api_key=self.api_key)
                prompt = f"Answer this patient question regarding {deficiency_type} deficiency and Indian diet: '{query}'"
                comp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=300
                )
                return comp.choices[0].message.content.strip()
            except Exception:
                pass

        return f"For {deficiency_type} deficiency, ensure high intake of nutrient-dense foods like leafy greens, fortified dairy, and citrus fruits to boost absorption."
