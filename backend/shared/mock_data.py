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
    SummaryInfo,
    BloodParameter,
    UploadedReportInfo,
    ModelGradeDetail,
)


def generate_mock_frontend_output(patient_id: str = "PAT-2026-8841") -> ModCFrontendOutput:
    """Generates a complete, schema-compliant sample payload."""
    return ModCFrontendOutput(
        generated_at=datetime.now(timezone.utc).isoformat(),
        schema_version="1.1",
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
                ],
                model_confirmed=True,
                etiology="IDA",
                model_confidence=0.71,
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
                ],
                model_confirmed=False,
            ),
        ],
        summary=SummaryInfo(
            flagged_deficiency_count=2,
            overall_risk_band="moderate"
        ),
        model=ModelGradeDetail(
            decision_code=1,
            etiology="IDA",
            na_count=0,
            complete_input=True,
            proba={"No Anemia": 0.08, "IDA": 0.71, "B12 Deficiency": 0.14, "Folate Deficiency": 0.07},
            proba_xgb={"No Anemia": 0.01, "IDA": 0.97, "B12 Deficiency": 0.01, "Folate Deficiency": 0.01},
            proba_gnn={"No Anemia": 0.15, "IDA": 0.45, "B12 Deficiency": 0.27, "Folate Deficiency": 0.13},
            alpha=0.5,
        ),
        model_confidence=0.71,
        blood_parameters=[
            BloodParameter(key="hemoglobin", name="Hemoglobin", value=10.4, unit="g/dL",
                           normal_range="12 - 15.5", status="Low", deviation=-0.165, model_input=True, confidence=0.95),
            BloodParameter(key="rbc", name="Total RBC Count", value=3.6, unit="million/cmm",
                           normal_range="4.2 - 5.4", status="Low", deviation=-0.143, model_input=True, confidence=0.92),
            BloodParameter(key="mcv", name="Mean Corpuscular Volume (MCV)", value=78.0, unit="fL",
                           normal_range="83 - 101", status="Low", deviation=-0.060, model_input=True, confidence=0.95),
            BloodParameter(key="ferritin", name="Serum Ferritin", value=11.2, unit="ng/mL",
                           normal_range="13 - 150", status="Low", deviation=-0.138, model_input=True, confidence=0.94),
            BloodParameter(key="iron", name="Serum Iron", value=52.0, unit="ug/dL",
                           normal_range="60 - 170", status="Low", deviation=-0.133, model_input=False, confidence=0.9),
            BloodParameter(key="tibc", name="Total Iron Binding Capacity (TIBC)", value=410.0, unit="ug/dL",
                           normal_range="250 - 450", status="Normal", deviation=0.0, model_input=False, confidence=0.9),
            BloodParameter(key="b12", name="Vitamin B12", value=372.0, unit="pg/mL",
                           normal_range="200 - 900", status="Normal", deviation=0.0, model_input=True, confidence=0.93),
            BloodParameter(key="folate", name="Serum Folate", value=9.8, unit="ng/mL",
                           normal_range="4.6 - 18.7", status="Normal", deviation=0.0, model_input=True, confidence=0.9),
            BloodParameter(key="wbc", name="Total Leucocyte Count (TLC)", value=6.8, unit="thousand/cmm",
                           normal_range="4 - 11", status="Normal", deviation=0.0, model_input=False, confidence=0.91),
            BloodParameter(key="platelet_count", name="Platelet Count", value=245.0, unit="lac/cmm",
                           normal_range="150 - 410", status="Normal", deviation=0.0, model_input=False, confidence=0.9),
        ],
        uploaded_report=UploadedReportInfo(
            filename="CBC_Blood_Report.pdf",
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            parse_confidence=0.93,
        ),
    )


if __name__ == "__main__":
    import json
    sample = generate_mock_frontend_output()
    print(json.dumps(sample.model_dump(), indent=2))
