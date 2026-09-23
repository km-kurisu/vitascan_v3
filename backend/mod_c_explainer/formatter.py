from datetime import datetime
from backend.shared.schemas import ModCFrontendOutput, ModB3Output, ModA3Output, PatientMetadata, DeficiencyDetail, SummarySection, KeyContributor, CrosscheckIndicator, DietRecommendation
from backend.mod_c_explainer.explainer import LLMExplainer
from backend.shared.storage import save_json_artifact

class ModCFormatter:
    def __init__(self):
        self.explainer = LLMExplainer()

    def format_pipeline_output(self, mod_b3: ModB3Output, mod_a3: ModA3Output = None) -> ModCFrontendOutput:
        patient_id = mod_b3.patient_id
        deficiencies = []
        flagged_count = 0
        worst_band = "none"

        band_order = {"severe": 3, "moderate": 2, "mild": 1, "none": 0}
        color_map = {"none": "#10b981", "mild": "#3b82f6", "moderate": "#f59e0b", "severe": "#ef4444"}

        for d_type, item in mod_b3.severity.items():
            if item.band != "none":
                flagged_count += 1
                if band_order.get(item.band, 0) > band_order.get(worst_band, 0):
                    worst_band = item.band

            exp_text = self.explainer.explain_deficiency(d_type, item.band, {"score": item.score})

            cross_ind = CrosscheckIndicator(available=False, agrees=False, source="none")
            if mod_a3 and d_type in mod_a3.crosscheck_signal:
                a3_item = mod_a3.crosscheck_signal[d_type]
                cross_ind = CrosscheckIndicator(
                    available=True,
                    agrees=a3_item.agrees_with_path_b,
                    source=a3_item.source
                )

            deficiencies.append(DeficiencyDetail(
                type=d_type,
                severity={"band": item.band, "score_pct": f"{int(item.score * 100)}%", "badge_color": color_map.get(item.band, "#3b82f6")},
                explanation=exp_text,
                key_contributors=[KeyContributor(biomarker="Serum Ferritin" if d_type=="iron" else d_type.title(), impact_pct=70.0)],
                crosscheck=cross_ind,
                diet_recommendations=[DietRecommendation(suggestion=f"Increase consumption of {d_type}-rich foods and follow FSSAI RDA guidelines.")]
            ))

        output = ModCFrontendOutput(
            generated_at=datetime.utcnow().isoformat() + "Z",
            schema_version="1.0",
            patient=PatientMetadata(patient_id=patient_id, age=32, gender="Female"),
            deficiencies=deficiencies,
            summary=SummarySection(flagged_deficiency_count=flagged_count, overall_risk_band=worst_band)
        )

        save_json_artifact("results", patient_id, output.model_dump())
        return output
