import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.shared.schemas import (
    ModB3Output,
    SeverityDetail,
    ModA3Output,
    CrosscheckDetail,
    ModCFrontendOutput
)
from backend.shared.mock_data import generate_mock_frontend_output


def test_mod_b3_schema_valid():
    data = {
        "patient_id": "TEST-101",
        "severity": {
            "iron": {"score": 0.71, "band": "moderate", "model": "gat"},
            "b12": {"score": 0.34, "band": "mild", "model": "gat"},
            "folate": {"score": 0.12, "band": "none", "model": "baseline"},
            "anemia": {"score": 0.42, "band": "mild", "model": "gat"}
        },
        "attention_weights": {
            "iron": {"ferritin": 0.62, "hemoglobin": 0.24, "tibc": 0.14}
        },
        "baseline_comparison": {
            "iron": {"gat": 0.71, "logistic_regression": 0.65, "random_forest": 0.69, "xgboost": 0.70}
        },
        "model_confidence": 0.88
    }
    model = ModB3Output(**data)
    assert model.patient_id == "TEST-101"
    assert model.severity["iron"].band == "moderate"


def test_mod_a3_schema_valid():
    data = {
        "patient_id": "TEST-101",
        "crosscheck_signal": {
            "anemia": {"confidence": 0.85, "source": "eyes", "agrees_with_path_b": True}
        },
        "visual_findings": ["Pallor of the conjunctiva", "Pale lower eyelid mucosa"],
        "model_confidence": 0.85
    }
    model = ModA3Output(**data)
    assert model.crosscheck_signal["anemia"].source == "eyes"
    assert model.visual_findings[0] == "Pallor of the conjunctiva"


def test_mock_frontend_output_valid():
    mock_payload = generate_mock_frontend_output("TEST-999")
    assert isinstance(mock_payload, ModCFrontendOutput)
    assert mock_payload.patient.patient_id == "TEST-999"
    assert len(mock_payload.deficiencies) > 0
