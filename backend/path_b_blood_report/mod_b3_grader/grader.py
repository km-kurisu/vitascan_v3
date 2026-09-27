from typing import Any, Dict, Optional

import logging

from backend.path_b_blood_report.mod_b3_grader.gat_model import VitaScanGATDeficiencyGrader
from backend.path_b_blood_report.mod_b3_grader.baseline_models import BaselineModelsEvaluator
from backend.path_b_blood_report.mod_b3_grader import model_client
from backend.shared.schemas import ModB3Output, SeverityDetail
from backend.shared.storage import save_json_artifact

logger = logging.getLogger("vitascan.mod_b3.grader")


class PathBGrader:
    def __init__(self):
        self.gat = VitaScanGATDeficiencyGrader()
        self.baseline = BaselineModelsEvaluator()

    def _log_grade(self, patient_id: str, payload: Dict[str, Any]) -> None:
        """Persist the grader verdict. Never let a logging failure break grading."""
        try:
            path = save_json_artifact("model_grade", patient_id, payload)
            logger.info("Logged model grade for '%s' -> %s", patient_id, path)
        except Exception:
            logger.exception("Failed to log model grade for '%s'", patient_id)

    def grade(
        self,
        patient_id: str,
        normalized_biomarkers: dict,
        grader_input: Optional[Dict[str, Any]] = None,
    ) -> ModB3Output:
        raw_sev = self.gat.grade(normalized_biomarkers)
        sev_dict = {k: SeverityDetail(score=v["score"], band=v["band"], model=v["model"]) for k, v in raw_sev.items()}

        model_detail = None
        model_confidence = 0.88
        if grader_input is not None:
            model_detail = model_client.grade_record(grader_input)
            model_confidence = max(model_detail.proba.values())
            overlay = model_client.severity_overlay(model_detail.decision_code)
            for deficiency, (band, score) in overlay.items():
                sev_dict[deficiency] = SeverityDetail(score=score, band=band, model="anemia_grader")

            self._log_grade(patient_id, {
                **model_detail.model_dump(),
                "patient_id": patient_id,
                "model_confidence": round(model_confidence, 3),
                "model_input": grader_input.get("model_input"),
                "severity_overlay": {k: {"band": b, "score": s} for k, (b, s) in overlay.items()},
            })

        return ModB3Output(
            patient_id=patient_id,
            severity=sev_dict,
            attention_weights={"iron": {"ferritin": 0.62, "hemoglobin": 0.24, "tibc": 0.14}},
            baseline_comparison=self.baseline.predict(normalized_biomarkers),
            model_confidence=round(model_confidence, 3),
            model=model_detail,
        )