import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from backend.path_b_blood_report.mod_b2_normalizer.composer import compose_pdf_raw
from backend.path_b_blood_report.mod_b2_normalizer.grader_input import GraderInputBuilder


def raw(filename, patient_id, biomarkers=None, raw_text=""):
    doc = {"filename": filename, "patient_id": patient_id, "raw_text": raw_text}
    if biomarkers is not None:
        doc["biomarkers"] = biomarkers
    return doc


def test_composes_biomarker_keys_first_file_wins():
    cbc = raw("cbc.pdf", "PAT-99", {
        "hemoglobin_hb": {"value": 12.5, "unit": "g/dL"},
        "total_rbc_count": {"value": 4.2, "unit": "million/cmm"},
        "mcv": {"value": 88.0, "unit": "fL"},
    })
    vitamins = raw("vitamins.pdf", "PAT-99", {
        "b12": {"value": 148.0, "unit": "pg/mL"},
        "total_rbc_count": {"value": 5.5, "unit": "million/cmm"},  # overridden
    })
    out = compose_pdf_raw([cbc, vitamins], patient_id="PAT-99")
    assert out["patient_id"] == "PAT-99"
    assert out["biomarkers"]["total_rbc_count"]["value"] == 4.2
    assert out["biomarkers"]["b12"]["value"] == 148.0
    assert len(out["composite_parts"]) == 2
    assert len(out["biomarker_provenance"]) == 4


def test_composite_build_resolves_across_parts_and_concatenates_text():
    cbc = raw("cbc.pdf", "PAT-99", {
        "hemoglobin_hb": {"value": "12.5 g/dL", "unit": "g/dL"},
        "total_rbc_count": {"value": 4.2, "unit": "million/cmm"},
    }, raw_text="Hemoglobin (Hb) 12.5 g/dL 13.0 - 17.0\nTotal RBC 4.2")
    vitamins = raw("vitamins.pdf", "PAT-99", {}, raw_text="Vitamin B12 148 pg/mL 197 - 771\nFolate 8.0 ng/mL")
    builder = GraderInputBuilder(imputer_joblib=None)
    rec = builder.build_composite([cbc, vitamins], patient_id="PAT-99")
    mi = rec["model_input"]
    assert mi["hemoglobin"] == 12.5
    assert mi["rbc"] == 4.2
    assert mi["b12"] == 148.0
    assert mi["folate"] == 8.0
    assert rec["composite"]["part_count"] == 2
    assert rec["composite"]["biomarker_provenance"][0]["filename"] == "cbc.pdf"
    assert rec["na_count"] == 5

    merged = compose_pdf_raw([cbc, vitamins], patient_id="PAT-99")
    assert "===== FILE: cbc.pdf (part 1/2) =====" in merged["raw_text"]
    assert "Folate 8.0 ng/mL" in merged["raw_text"]


def test_composer_requires_at_least_one_doc():
    try:
        compose_pdf_raw([], patient_id="P")
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty docs")


def test_duplicate_same_key_across_parts_first_wins_and_build_uses_best_alias():
    a = raw("a.pdf", "PAT-7", {
        "rbc": {"value": 300.0, "unit": ""},                 # junk value
        "total_rbc_count": {"value": 6.1, "unit": "million/cmm"},  # bad (outside range)
    })
    b = raw("b.pdf", "PAT-7", {
        "total_rbc_count": {"value": 4.9, "unit": "million/cmm"},
    })
    out = compose_pdf_raw([a, b])
    assert out["biomarkers"]["total_rbc_count"]["value"] == 6.1  # first file wins at key level
    rec = GraderInputBuilder(imputer_joblib=None).build(out)
    assert rec["model_input"]["rbc"] == 6.1