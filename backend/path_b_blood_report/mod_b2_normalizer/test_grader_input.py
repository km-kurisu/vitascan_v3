import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from backend.path_b_blood_report.mod_b2_normalizer.grader_input import (
    GraderInputBuilder,
    build_feature_matrix,
    parse_number,
)


def make_raw(patient_id="PAT-T", biomarkers=None, raw_text=""):
    base = {"patient_id": patient_id, "filename": f"{patient_id}.pdf", "raw_text": raw_text}
    if biomarkers is not None:
        base["biomarkers"] = biomarkers
    return base


def test_structured_dict_resolution_and_payload():
    raw = make_raw(biomarkers={
        "hemoglobin": {"value": 14.5, "unit": "g/dL"},
        "mcv": {"value": 90.3, "unit": "fL"},
        "rbc": {"value": 4.79, "unit": "million/uL"},
        "mch": {"value": 30.2, "unit": "pg"},
        "mchc": {"value": 33.4, "unit": "g/dL"},
    })
    rec = GraderInputBuilder(imputer_joblib=None).build(raw)
    mi = rec["model_input"]
    assert mi["hemoglobin"] == 14.5
    assert mi["mcv"] == 90.3
    assert mi["rbc"] == 4.79
    assert mi["mchc"] == 33.4
    assert mi["b12"] is None and mi["folate"] is None and mi["ferritin"] is None
    assert rec["na_count"] == 4
    payload = GraderInputBuilder().to_payload(rec)
    assert payload["patient_id"] == "PAT-T"
    assert set(payload.keys()) == {
        "patient_id", "hemoglobin", "mcv", "rdw", "ferritin", "b12",
        "folate", "rbc", "mch", "mchc",
    }


def test_alias_preferred_over_junk_keys():
    raw = make_raw(biomarkers={
        "rbc": {"value": 300.0, "unit": ""},       # GLiNER garbage
        "total_rbc_count": {"value": 5.2, "unit": "million/cmm"},
        "83": {"value": 101.0, "unit": ""},          # junk ref-range key
    })
    rec = GraderInputBuilder(imputer_joblib=None).build(raw)
    assert rec["model_input"]["rbc"] == 5.2
    assert rec["biomarkers"]["rbc"]["resolved_from"] == "total_rbc_count"


def test_non_numeric_token_becomes_null():
    raw = make_raw(biomarkers={
        "folate": {"value": "Ferritin", "unit": "ng/mL"},   # GLiNER mislabel
        "ferritin": {"value": 32.0, "unit": "ng/mL"},
    })
    rec = GraderInputBuilder(imputer_joblib=None).build(raw)
    assert rec["model_input"]["folate"] is None
    assert rec["model_input"]["ferritin"] == 32.0
    assert "non-numeric" in rec["biomarkers"]["folate"]["status"]


def test_unit_conversion():
    raw = make_raw(biomarkers={
        "b12": {"value": 0.3, "unit": "ng/mL"},   # -> 300 pg/mL
        "ferritin": {"value": 30.2, "unit": "ug/L"},  # -> 30.2 ng/mL
    })
    rec = GraderInputBuilder(imputer_joblib=None).build(raw)
    assert rec["model_input"]["b12"] == 300.0
    assert rec["model_input"]["ferritin"] == 30.2


def test_implausible_value_rejected():
    raw = make_raw(biomarkers={
        "hemoglobin": {"value": 103, "unit": "g/dL"},
        "mcv": {"value": 1.2, "unit": "fL"},     # outside soft band
    })
    rec = GraderInputBuilder(imputer_joblib=None).build(raw)
    assert rec["model_input"]["hemoglobin"] is None
    assert rec["model_input"]["mcv"] is None


def test_raw_text_single_line_row():
    text = "Hemoglobin (Hb) 12.5 Low 13.0 - 17.0 g/dL\n\nWBC Count 9.4\nESR 8 mm/hr"
    rec = GraderInputBuilder(imputer_joblib=None).build(make_raw(raw_text=text))
    assert rec["model_input"]["hemoglobin"] == 12.5


def test_raw_text_columnar_block():
    text = ("Hemoglobin\ng/dL\n13.0 - 16.5\nColorimetric\n14.5\n\n"
            "RBC Count\nmillion/cmm\n4.5 - 5.5\nElectrical impedance\n4.79")
    rec = GraderInputBuilder(imputer_joblib=None).build(make_raw(raw_text=text))
    assert rec["model_input"]["hemoglobin"] == 14.5
    assert rec["model_input"]["rbc"] == 4.79


def test_raw_text_blank_line_separated_cells():
    text = ("\n\nVitamin B12\nL\npg/mL\n187 - 833\nCLIA\n< 148\n\n"
            "Vitamin B12 is essential in DNA synthesis.\n\n"
            "It is recommended to test for vitamin B12 or folate deficiency.\n")
    rec = GraderInputBuilder(imputer_joblib=None).build(make_raw(raw_text=text))
    assert rec["model_input"]["b12"] == 148.0
    assert rec["biomarkers"]["b12"]["resolved_from"] == "raw_text"


def test_hba1c_not_misread_as_hemoglobin():
    text = "Glycosylated Hemoglobin (HbA1c): 6.0 %\nGlucose - Fasting: 101 mg/dL"
    rec = GraderInputBuilder(imputer_joblib=None).build(make_raw(raw_text=text))
    assert rec["model_input"]["hemoglobin"] is None


def test_prose_mentions_ignored():
    text = ("A normal serum concentration of B12 does not rule out deficiency of "
            "vitamin B12. The assay for MMA is the most sensitive test.")
    rec = GraderInputBuilder(imputer_joblib=None).build(make_raw(raw_text=text))
    assert rec["model_input"]["b12"] is None


def test_derived_indices_and_feature_matrix():
    raw = make_raw(biomarkers={
        "mch": {"value": 30.2, "unit": "pg"},
        "rbc": {"value": 4.79, "unit": "million/uL"},
        "mcv": {"value": 90.3, "unit": "fL"},
        "mchc": {"value": 33.4, "unit": "g/dL"},
    })
    rec = GraderInputBuilder(imputer_joblib=None).build(raw)
    d = rec["derived"]
    assert abs(d["srivastava_index"] - 30.2 / 4.79) < 1e-2
    assert abs(d["mentzer_index"] - 90.3 / 4.79) < 1e-2
    assert abs(d["cell_hb_density"] - 30.2 * 33.4 / 100.0) < 1e-2
    mat = build_feature_matrix([rec["model_input"], {"hemoglobin": None, "mcv": None,
                                                      "rdw": None, "ferritin": None,
                                                      "b12": None, "folate": None,
                                                      "rbc": None, "mch": None, "mchc": None}])
    assert mat.shape == (2, 12)
    assert np.isnan(mat[1, 0])


def test_parse_number_edge_cases():
    assert parse_number(" < 0.1 ") == 0.1
    assert parse_number("1,234.5") == 1234.5
    assert parse_number("Negative") is None
    assert parse_number("--") is None
    assert parse_number(None) is None