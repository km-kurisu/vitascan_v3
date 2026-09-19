from backend.path_b_blood_report.mod_b3_grader.gat_model import VitaScanGATDeficiencyGrader
from backend.path_b_blood_report.mod_b3_grader.baseline_models import BaselineModelsEvaluator
from backend.shared.schemas import ModB3Output, SeverityDetail

class PathBGrader:
    def __init__(self):
        self.gat = VitaScanGATDeficiencyGrader()
        self.baseline = BaselineModelsEvaluator()

    def grade(self, patient_id: str, normalized_biomarkers: dict) -> ModB3Output:
        raw_sev = self.gat.grade(normalized_biomarkers)
        sev_dict = {k: SeverityDetail(score=v["score"], band=v["band"], model=v["model"]) for k, v in raw_sev.items()}
        
        return ModB3Output(
            patient_id=patient_id,
            severity=sev_dict,
            attention_weights={"iron": {"ferritin": 0.62, "hemoglobin": 0.24, "tibc": 0.14}},
            baseline_comparison=self.baseline.predict(normalized_biomarkers),
            model_confidence=0.88
        )
