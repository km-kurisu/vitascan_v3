import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.path_a_symptom_image.groq_vision_analyzer import GroqSymptomVisionAnalyzer
from backend.path_a_symptom_image.crosscheck import PathACrosscheckSignal


def test_fallback_analysis_shape():
    analyzer = GroqSymptomVisionAnalyzer()
    res = analyzer._fallback_analysis("eyes")
    assert res["deficiency"] == "anemia"
    assert 0.0 <= res["confidence"] <= 1.0
    assert isinstance(res["visual_findings"], list)
    assert res["visual_findings"]
    assert res["source_region"] == "eyes"


def test_crosscheck_signal_maps_findings():
    analysis = {
        "deficiency": "b12",
        "confidence": 0.73,
        "visual_findings": ["Pale conjunctiva detected", "Smooth tongue surface"],
        "agrees_with_path_b": True,
        "source_region": "eyes"
    }
    signal = PathACrosscheckSignal().format_signal("PAT-A3-1001", analysis)
    dumped = signal.model_dump()

    assert dumped["patient_id"] == "PAT-A3-1001"
    assert "b12" in dumped["crosscheck_signal"]
    assert dumped["crosscheck_signal"]["b12"]["source"] == "eyes"
    assert dumped["crosscheck_signal"]["b12"]["confidence"] == 0.73
    assert dumped["crosscheck_signal"]["b12"]["agrees_with_path_b"] is True
    assert dumped["visual_findings"] == ["Pale conjunctiva detected", "Smooth tongue surface"]
    assert dumped["model_confidence"] == 0.73


def test_crosscheck_signal_defaults():
    analysis = {
        "deficiency": "anemia",
        "confidence": 0.5,
        "visual_findings": [],
        "agrees_with_path_b": True,
        "source_region": "nails"
    }
    signal = PathACrosscheckSignal().format_signal("PAT-A3-1002", analysis)
    dumped = signal.model_dump()
    assert dumped["visual_findings"] == []
    assert dumped["crosscheck_signal"]["anemia"]["source"] == "nails"