"""
Automated Unit Tests for OCR Extraction Logging, Shared Folder JSON Storage,
and BioBERT Biomarker Extraction with Per-Patient Storage.
"""
import os
import sys
import json
import logging
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.shared.storage import save_extracted_json, save_patient_biomarkers_json, EXTRACTIONS_DIR, PATIENT_BIOMARKERS_DIR
from backend.path_b_blood_report.mod_b1_extractor.extractor import BloodReportExtractor
from backend.path_b_blood_report.mod_b1_extractor.biobert_extractor import BioBERTBiomarkerExtractor


def test_save_extracted_json_logging_and_storage(caplog):
    """Verifies that extracted OCR JSON is saved in backend/shared/extractions/ with logger output and patient details."""
    caplog.set_level(logging.INFO)
    
    patient_id = "PAT-TEST-1001"
    sample_data = {
        "filename": "sample_cbc_report.pdf",
        "raw_text": "Patient Name: Alice Smith\nAge: 34\nGender: Female\nDate: 2026-08-19\nSerum Ferritin: 11.2 ng/mL\nHemoglobin: 10.4 g/dL\nVitamin B12: 210 pg/mL",
        "patient_details": {
            "patient_id": patient_id,
            "name": "Alice Smith",
            "age": "34",
            "gender": "Female",
            "date": "2026-08-19"
        },
        "biomarkers": {
            "ferritin": {"canonical_name": "Serum Ferritin", "value": 11.2, "unit": "ng/mL"},
            "hemoglobin": {"canonical_name": "Hemoglobin", "value": 10.4, "unit": "g/dL"}
        },
        "extraction_method": "PyMuPDF text extraction",
        "char_count": 145
    }

    file_path = save_extracted_json(patient_id, sample_data)
    
    assert os.path.exists(file_path)
    assert file_path.startswith(str(EXTRACTIONS_DIR))
    
    with open(file_path, "r", encoding="utf-8") as f:
        saved_json = json.load(f)

    assert saved_json["patient_details"]["patient_id"] == patient_id
    assert saved_json["filename"] == "sample_cbc_report.pdf"
    assert saved_json["patient_details"]["name"] == "Alice Smith"
    assert "ferritin" in saved_json["biomarkers"]
    assert "Serum Ferritin" in saved_json["raw_text"]
    assert "Saved pdf_raw JSON" in caplog.text


def test_biobert_biomarker_extractor_and_patient_saving(caplog):
    """Verifies BioBERT biomarker extraction and saving for a patient in backend/shared/patient_biomarkers/."""
    caplog.set_level(logging.INFO)

    patient_id = "PAT-BIOBERT-2002"
    sample_report = (
        "LABORATORY DIAGNOSTICS REPORT\n"
        "Patient Name: John Doe\n"
        "Age: 45 / Male\n"
        "Serum Ferritin: 11.2 ng/mL (Ref: 13 - 150)\n"
        "Hemoglobin: 10.4 g/dL (Ref: 12.0 - 15.5)\n"
        "Vitamin B12: 210 pg/mL (Ref: 200 - 900)\n"
        "Folate: 4.8 ng/mL (Ref: 4.6 - 18.7)\n"
        "Platelets: 250000 cells/uL\n"
    )

    extractor = BioBERTBiomarkerExtractor()
    res = extractor.extract_and_save_biomarkers(sample_report, patient_id=patient_id)

    assert "biomarkers" in res
    assert len(res["biomarkers"]) > 0
    assert "ferritin" in res["biomarkers"]
    assert "platelets" in res["biomarkers"]
    assert res["biomarkers"]["ferritin"]["value"] == 11.2

    # Check patient biomarker JSON file creation
    patient_file = PATIENT_BIOMARKERS_DIR / f"{patient_id}_latest.json"
    assert os.path.exists(patient_file)

    with open(patient_file, "r", encoding="utf-8") as f:
        saved_biomarkers = json.load(f)

    assert saved_biomarkers["patient_details"]["patient_id"] == patient_id
    assert saved_biomarkers["patient_details"]["name"] == "John Doe"
    assert "ferritin" in saved_biomarkers["biomarkers"]
    assert "Starting Biomedical NER biomarker extraction" in caplog.text
    assert "Successfully extracted and saved BioBERT biomarkers" in caplog.text


def test_blood_report_extractor_with_logging_and_storage(caplog):
    """Verifies BloodReportExtractor logs extraction steps, extracts patient details & biomarkers, and calls JSON saving."""
    caplog.set_level(logging.INFO)

    patient_id = "PAT-EXTRACTOR-3003"
    extractor = BloodReportExtractor()
    sample_bytes = (
        b"Patient Name: Sarah Connor\nAge: 29 / Female\nDate: 2026-08-19\n"
        b"Serum Ferritin: 11.2 ng/mL\nHemoglobin: 10.4 g/dL\nVitamin B12: 210 pg/mL\n"
        b"Serum Creatinine: 0.8 mg/dL\n"
    )

    result = extractor.extract_from_bytes(sample_bytes, "test_report.txt", patient_id=patient_id, save_json=True)

    assert result["patient_id"] == patient_id
    assert result["patient_details"]["name"] == "Sarah Connor"
    assert result["patient_details"]["gender"] == "Female"
    assert "ferritin" in result["biomarkers"]
    assert "creatinine" in result["biomarkers"]
    assert "json_storage_path" in result
    assert os.path.exists(result["json_storage_path"])

    with open(result["json_storage_path"], "r", encoding="utf-8") as f:
        extraction_file_content = json.load(f)

    assert extraction_file_content["patient_details"]["name"] == "Sarah Connor"
    assert "ferritin" in extraction_file_content["biomarkers"]
    assert "Starting blood report extraction" in caplog.text
    assert "Blood report extraction complete" in caplog.text


def test_process_and_save_functionality(tmp_path):
    """Verifies process_and_save key function on BloodReportExtractor."""
    extractor = BloodReportExtractor()
    sample_file = tmp_path / "test_report.txt"
    sample_file.write_text(
        "PATIENT REPORT\n"
        "Patient Name: Mark Taylor\n"
        "Age: 42\n"
        "Gender: Male\n"
        "Glycosylated Hemoglobin (HbA1c): 6.0 %\n"
        "Glucose - Fasting: 101 mg/dL\n"
        "Serum Creatinine: 0.79 mg/dL\n",
        encoding="utf-8"
    )

    res = extractor.process_and_save(str(sample_file))
    assert res["status"] == "ok"
    assert "result" in res
    assert res["result"]["age"] == "42"
    assert res["result"]["gender"] == "Male"
    assert "labs" in res["result"]
    assert os.path.exists(res["output_file"])


def test_gliner_biobert_hybrid_source_mix():
    """Verifies GLiNER / BioBERT hybrid biomedical NER parser and source_mix metadata."""
    extractor = BioBERTBiomarkerExtractor()
    sample_report = (
        "Patient Name: Emily Watson\n"
        "Age: 38 / Female\n"
        "HbA1c: 5.8 %\n"
        "Serum Ferritin: 12.0 ng/mL\n"
        "Glucose - Fasting: 98 mg/dL\n"
    )

    result = extractor.extract_biomarkers(sample_report)
    assert "biomarkers" in result
    assert "source_mix" in result
    assert result["source_mix"] in ["regex_only", "gliner_only", "hybrid"]
    assert "parse_confidence" in result
    assert result["parse_confidence"] > 0.0

