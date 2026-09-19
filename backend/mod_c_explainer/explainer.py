import os
import logging
from typing import Dict, Any

logger = logging.getLogger("vitascan.explainer")

class LLMExplainer:
    def __init__(self):
        self.api_key = os.getenv("GROQ_RAG_API_KEY") or os.getenv("GROQ_API_KEY")
        self.client = None
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except Exception:
                pass

    def explain_deficiency(self, deficiency_type: str, severity_band: str, biomarkers: Dict[str, Any]) -> str:
        if self.client:
            try:
                prompt = (
                    f"Explain a {severity_band} {deficiency_type} deficiency to a patient in clear, compassionate plain English. "
                    f"Key Biomarker lab values: {biomarkers}. "
                    "Keep explanation under 3 concise paragraphs."
                )
                comp = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=250,
                    temperature=0.3
                )
                return comp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Groq explainer fallback: {e}")

        return (
            f"Your report indicates a {severity_band} level of {deficiency_type} deficiency. "
            "Biomarkers show deviations from standard reference ranges. Incorporating recommended dietary adjustments "
            "and consulting a physician is advised."
        )
