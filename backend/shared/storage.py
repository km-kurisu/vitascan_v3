import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("vitascan.storage")

BASE_STORAGE_DIR = Path(__file__).parent / "extractions"

SUBDIRS = {
    "pdf_raw": BASE_STORAGE_DIR / "pdf_raw",
    "biomarkers": BASE_STORAGE_DIR / "biomarkers",
    "groq_symptom": BASE_STORAGE_DIR / "groq_symptom",
    "results": BASE_STORAGE_DIR / "results",
    "diet_plans": BASE_STORAGE_DIR / "diet_plans",
    "reminders": BASE_STORAGE_DIR / "reminders",
}

for d in SUBDIRS.values():
    d.mkdir(parents=True, exist_ok=True)

EXTRACTIONS_DIR = BASE_STORAGE_DIR
PATIENT_BIOMARKERS_DIR = SUBDIRS["biomarkers"]

def save_json_artifact(category: str, patient_id: str, data: Dict[str, Any]) -> str:
    """Saves an organized JSON artifact under shared/extractions/<category>/"""
    if category not in SUBDIRS:
        subdir = BASE_STORAGE_DIR / category
        subdir.mkdir(parents=True, exist_ok=True)
        SUBDIRS[category] = subdir

    timestamp = int(time.time())
    file_path = SUBDIRS[category] / f"{patient_id}_{timestamp}_{category}.json"
    latest_path = SUBDIRS[category] / f"{patient_id}_latest.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {category} JSON for patient '{patient_id}' -> {file_path}")
    return str(file_path)

def load_latest_artifact(category: str, patient_id: str) -> Optional[Dict[str, Any]]:
    """Loads the latest saved JSON artifact for a given category and patient_id."""
    if category not in SUBDIRS:
        return None

    latest_path = SUBDIRS[category] / f"{patient_id}_latest.json"
    if not latest_path.exists():
        return None

    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading artifact {latest_path}: {e}")
        return None

# Compatibility aliases for legacy modules
def save_patient_biomarkers_json(patient_id: str, data: Dict[str, Any]) -> str:
    return save_json_artifact("biomarkers", patient_id, data)

def save_patient_extraction_json(patient_id: str, data: Dict[str, Any]) -> str:
    return save_json_artifact("pdf_raw", patient_id, data)


def save_extracted_json(patient_id: str, data: Dict[str, Any]) -> str:
    return save_json_artifact("pdf_raw", patient_id, data)
