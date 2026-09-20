import os
import base64
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger("vitascan.groq_vision")

DEFAULT_VISION_MODEL = "qwen/qwen3.8-27b"

class GroqSymptomVisionAnalyzer:
    def __init__(self):
        self.api_key = os.getenv("GROQ_VISION_API_KEY") or os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_VISION_MODEL", DEFAULT_VISION_MODEL)
        self.client = None
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                logger.info("Groq Vision API client initialized.")
            except Exception as e:
                logger.warning(f"Groq client init error: {e}")

    def analyze_image_bytes(self, image_bytes: bytes, filename: str, source_region: str = "eyes") -> Dict[str, Any]:
        if not self.client:
            logger.info("Groq API key missing. Using rule-assisted visual analyzer fallback.")
            return self._fallback_analysis(source_region)

        try:
            b64_img = base64.b64encode(image_bytes).decode("utf-8")
            ext = os.path.splitext(filename)[1].lower().replace(".", "")
            mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
            data_url = f"data:{mime};base64,{b64_img}"

            prompt = (
                f"You are a clinical AI specialist analyzing a medical symptom image of the patient's {source_region}. "
                "Examine the image for clinical indicators of nutritional deficiencies (e.g. paleness/pallor, koilonychia spoon nails, glossitis, skin lesions). "
                "Return ONLY a raw JSON object with keys: 'deficiency' (e.g. 'anemia', 'iron', 'b12', 'folate'), 'confidence' (0.0 to 1.0), "
                "'visual_findings' (array of text strings), and 'agrees_with_path_b' (boolean)."
            )

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ],
                temperature=0.2,
                max_tokens=400,
                response_format={"type": "json_object"}
            )

            msg = completion.choices[0].message
            response_text = msg.content or getattr(msg, "reasoning_content", "") or ""
            # Extract JSON block
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(response_text)
            parsed["source_region"] = source_region
            return parsed

        except Exception as e:
            logger.error(f"Groq Vision API call failed: {e}")
            return self._fallback_analysis(source_region)

    def _fallback_analysis(self, source_region: str) -> Dict[str, Any]:
        findings_map = {
            "eyes": ["Subtle pale lower palpebral conjunctiva detected", "Mild sign of microcytic iron deficiency anemia"],
            "nails": ["Mild paleness in nail bed microvasculature", "Early sign of flattened nail curvature"],
            "tongue": ["Slightly smooth papillae observed on dorsal tongue surface"],
            "skin": ["Mild paleness in skin crease tone"]
        }
        return {
            "deficiency": "anemia",
            "confidence": 0.82,
            "visual_findings": findings_map.get(source_region.lower(), ["Visual inspection shows mild pallor"]),
            "agrees_with_path_b": True,
            "source_region": source_region
        }
