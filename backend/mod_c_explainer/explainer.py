import os
import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("vitascan.explainer")

# Tried in order; the first model the key can actually reach wins and is cached
# for the process (llama-3.3-70b-versatile was retired from Groq's catalogue).
DEFAULT_MODELS: Tuple[str, ...] = (
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
)


# gpt-oss spends part of the budget on reasoning, so 250 truncates mid-sentence.
MAX_TOKENS = 800

_CITATION_RE = re.compile(r"[\u3010\u3011]\d+[\u3010\u3011]?|\(\d+\)")


def _as_plain_text(content: Any) -> str:
    """Reasoning-tuned models answer in markdown with citation markers; the
    results page renders plain paragraphs, so strip both."""
    if not content:
        return ""
    text = str(content)
    text = _CITATION_RE.sub("", text)
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"(?m)^\s*[-*#]+\s*", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


class LLMExplainer:
    def __init__(self):
        self.api_key = os.getenv("GROQ_RAG_API_KEY") or os.getenv("GROQ_API_KEY")
        override = (os.getenv("GROQ_EXPLAINER_MODEL") or "").strip()
        self.models: List[str] = [override] if override else list(DEFAULT_MODELS)
        self._working_model: str | None = None
        self.client = None
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except Exception:
                pass

    def _ask(self, prompt: str) -> str | None:
        """First model that answers wins; retired model ids are skipped."""
        if not self.client:
            return None
        order = [self._working_model] if self._working_model else []
        order += [m for m in self.models if m not in order]
        for model in order:
            try:
                comp = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_TOKENS,
                    temperature=0.3,
                )
                text = _as_plain_text(comp.choices[0].message.content)
                if text:
                    self._working_model = model
                    return text
            except Exception as e:
                logger.warning(f"Groq explainer model {model} unavailable: {str(e)[:120]}")
        return None

    def explain_deficiency(
        self, deficiency_type: str, severity_band: str, biomarkers: Dict[str, Any]
    ) -> str:
        measured = self._format_measured(biomarkers)
        answer = self._ask(
            f"Explain a {severity_band} {deficiency_type} deficiency to a patient in clear, "
            f"compassionate plain English. These are the only lab values available: {measured}. "
            "Cite only these values, never invent another number or reference range. "
            "If a value is unavailable, say the report did not include it. "
            "Keep the explanation under 3 concise paragraphs. "
            "Plain text only: no markdown, no bold, no bullet points, no citation markers."
        )
        if answer:
            return answer
        return self._fallback(deficiency_type, severity_band, biomarkers)

    @staticmethod
    def _format_measured(biomarkers: Dict[str, Any]) -> str:
        if not biomarkers:
            return "none available for this report"
        return "; ".join(
            f"{item.get('name', key)} {item.get('value', '')} "
            f"(reference {item.get('range', 'unknown')}, {item.get('status', 'unknown')})"
            for key, item in biomarkers.items()
        )

    @staticmethod
    def _fallback(
        deficiency_type: str, severity_band: str, biomarkers: Dict[str, Any]
    ) -> str:
        """Used with no API key or no reachable model. Stays factual: it only
        ever repeats values that were actually measured."""
        measured = LLMExplainer._format_measured(biomarkers)
        if biomarkers:
            return (
                f"Your report indicates a {severity_band} level of {deficiency_type} deficiency, "
                f"based on the following measured values: {measured}. "
                "Incorporating recommended dietary adjustments and consulting a physician is advised."
            )
        return (
            f"Your report indicates a {severity_band} level of {deficiency_type} deficiency. "
            "The report did not include the marker values this grading is based on, so the "
            "result should be confirmed with a physician."
        )
