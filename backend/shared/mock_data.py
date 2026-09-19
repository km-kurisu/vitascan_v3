"""
Mock data generator for frontend and pipeline development.
Returns realistic modC_frontend_output.json payloads.
"""
from datetime import datetime, timezone
from backend.shared.schemas import (
    ModCFrontendOutput,
    PatientInfo,
    DeficiencyItem,
    KeyContributor,
    FrontendCrosscheck,
    DietRecommendation,
    SummaryInfo
)


def generate_mock_frontend_output(patient_id: str = "PAT-2026-8841") -> ModCFrontendOutput:
    """Generates a complete, schema-compliant sample payload."""
    return ModCFrontendOutput(
        generated_at=datetime.now(timezone.utc).isoformat(),
        schema_version="1.0",
        patient=PatientInfo(
            patient_id=patient_id,
            age=28,
            gender="Female"
        ),
        deficiencies=[
            DeficiencyItem(
                type="iron",
                severity={
                    "band": "moderate",
                    "score_pct": "71.0%",
                    "badge_color": "#f59e0b"
                },
                explanation=(
                    "Graph Attention analysis indicates moderate Iron Deficiency Risk. "
                    "Serum Ferritin level (11.2 ng/mL) is significantly below the normal range, "
                    "supported by lower Hemoglobin (10.4 g/dL). Attention weights assign 62% impact to Ferritin."
                ),
                key_contributors=[
                    KeyContributor(biomarker="Serum Ferritin", impact_pct=62.0, direction="negative"),
                    KeyContributor(biomarker="Hemoglobin", impact_pct=24.0, direction="negative"),
                    KeyContributor(biomarker="Total Iron Binding Capacity (TIBC)", impact_pct=14.0, direction="positive")
                ],
                crosscheck=FrontendCrosscheck(
                    available=True,
                    agrees=True,
                    source="eyes (conjunctival pallor)"
                ),
                diet_recommendations=[
                    DietRecommendation(suggestion="Increase consumption of dark leafy greens (spinach, amaranth), jaggery, and legumes.", fssai_checked=True),
                    DietRecommendation(suggestion="Pair iron-rich meals with Vitamin C (lemon juice, amla) to enhance non-heme iron absorption.", fssai_checked=True),
                    DietRecommendation(suggestion="Avoid drinking tea or coffee immediately before or after meals as tannins inhibit iron absorption.", fssai_checked=True)
                ]
            ),
            DeficiencyItem(
                type="b12",
                severity={
                    "band": "mild",
                    "score_pct": "34.0%",
                    "badge_color": "#3b82f6"
                },
                explanation=(
                    "Serum B12 is borderline (210 pg/mL). GAT graph analysis shows mild secondary influence on overall red cell indices (MCV 98 fL)."
                ),
                key_contributors=[
                    KeyContributor(biomarker="Serum B12", impact_pct=75.0, direction="negative"),
                    KeyContributor(biomarker="Mean Corpuscular Volume (MCV)", impact_pct=25.0, direction="positive")
                ],
                crosscheck=FrontendCrosscheck(
                    available=True,
                    agrees=True,
                    source="tongue (glossitis/smooth tongue)"
                ),
                diet_recommendations=[
                    DietRecommendation(suggestion="Include fortified dairy products, milk, paneer, and curd in daily diet.", fssai_checked=True),
                    DietRecommendation(suggestion="Consider consulting a physician for oral Cyanocobalamin supplementation if dietary intake is strictly plant-based.", fssai_checked=True)
                ]
            ),
            DeficiencyItem(
                type="anemia",
                severity={
                    "band": "mild",
                    "score_pct": "42.0%",
                    "badge_color": "#3b82f6"
                },
                explanation=(
                    "Mild anemia detected primarily driven by Iron deficiency rather than Folate or Vitamin B12 depletion."
                ),
                key_contributors=[
                    KeyContributor(biomarker="Hemoglobin", impact_pct=68.0, direction="negative"),
                    KeyContributor(biomarker="Hematocrit", impact_pct=32.0, direction="negative")
                ],
                crosscheck=FrontendCrosscheck(
                    available=False,
                    agrees=False,
                    source="none"
                ),
                diet_recommendations=[
                    DietRecommendation(suggestion="Follow the Iron deficiency nutritional guidance and monitor Hemoglobin levels after 4-6 weeks.", fssai_checked=True)
                ]
            )
        ],
        summary=SummaryInfo(
            flagged_deficiency_count=3,
            overall_risk_band="moderate"
        )
    )


if __name__ == "__main__":
    import json
    sample = generate_mock_frontend_output()
    print(json.dumps(sample.model_dump(), indent=2))
