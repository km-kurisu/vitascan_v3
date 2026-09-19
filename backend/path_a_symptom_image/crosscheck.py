from typing import Dict, Any
from backend.shared.schemas import ModA3Output, CrosscheckItem

class PathACrosscheckSignal:
    def format_signal(self, patient_id: str, groq_analysis: Dict[str, Any]) -> ModA3Output:
        deficiency_type = groq_analysis.get("deficiency", "anemia")
        confidence = float(groq_analysis.get("confidence", 0.85))
        source = groq_analysis.get("source_region", "eyes")
        findings = groq_analysis.get("visual_findings", ["Pallor observed in symptom photograph"])
        agrees = bool(groq_analysis.get("agrees_with_path_b", True))

        signal_dict = {
            deficiency_type: CrosscheckItem(
                confidence=confidence,
                source=source,
                agrees_with_path_b=agrees
            )
        }

        return ModA3Output(
            patient_id=patient_id,
            crosscheck_signal=signal_dict,
            visual_findings=findings,
            model_confidence=confidence
        )
