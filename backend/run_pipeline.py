import sys
import os

# Add root directory to sys.path so 'import backend...' works from inside backend/ directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from datetime import datetime, timezone
import time
import os
import logging
from dotenv import load_dotenv

load_dotenv()

from backend.path_b_blood_report.mod_b1_extractor.extractor import BloodReportExtractor
from backend.path_b_blood_report.mod_b1_extractor.biobert_extractor import BioBERTBiomarkerExtractor
from backend.path_b_blood_report.mod_b2_normalizer.normalizer import BiomarkerNormalizer
from backend.path_b_blood_report.mod_b3_grader.grader import PathBGrader
from backend.path_a_symptom_image.groq_vision_analyzer import GroqSymptomVisionAnalyzer
from backend.path_a_symptom_image.crosscheck import PathACrosscheckSignal
from backend.mod_c_explainer.formatter import ModCFormatter

from backend.reminders.schemas import Reminder, ReminderCreate
from backend.reminders.reminders_service import (
    create_reminder,
    list_reminders,
    delete_reminder,
    process_due_reminders,
)

from backend.diet_rag_service.diet_router import router as diet_router
from backend.shared.mock_data import generate_mock_frontend_output
from backend.shared.storage import save_json_artifact, load_latest_artifact

logger = logging.getLogger("vitascan.pipeline")

app = FastAPI(title="VitaScan V3 Orchestrator", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(diet_router)

b1_extractor = BloodReportExtractor()
biobert_extractor = BioBERTBiomarkerExtractor()
normalizer = BiomarkerNormalizer()
b3_grader = PathBGrader()

groq_vision = GroqSymptomVisionAnalyzer()
a3_crosscheck = PathACrosscheckSignal()
mod_c_formatter = ModCFormatter()

pipeline_state = {"latest_result": None}

@app.get("/")
def root():
    return {"status": "ok", "app": "VitaScan V3 Orchestrator (Groq API Edition)"}

@app.get("/status")
def get_status():
    return {
        "status": "ready",
        "extractor_mode": biobert_extractor.GLINER_MODEL_NAME if biobert_extractor.is_gliner_active else "regex",
        "groq_vision": "active" if groq_vision.client else "fallback_mode"
    }

@app.post("/upload-report")
async def upload_report(file: UploadFile = File(...), patient_id: Optional[str] = Form("PAT-DEMO123")):
    content = await file.read()
    raw_extraction = b1_extractor.extract_from_bytes(content, file.filename, patient_id)
    save_json_artifact("pdf_raw", patient_id, raw_extraction)

    ner_res = biobert_extractor.extract_biomarkers(raw_extraction["raw_text"])
    save_json_artifact("biomarkers", patient_id, ner_res)

    norm_res = normalizer.normalize(ner_res["biomarkers"])
    b3_output = b3_grader.grade(patient_id, norm_res)

    formatted = mod_c_formatter.format_pipeline_output(b3_output)
    pipeline_state["latest_result"] = formatted.model_dump()
    return formatted.model_dump()

@app.post("/upload-symptom-photo")
async def upload_symptom_photo(
    file: UploadFile = File(...),
    source_region: str = Form("eyes"),
    patient_id: Optional[str] = Form("PAT-DEMO123")
):
    content = await file.read()
    groq_res = groq_vision.analyze_image_bytes(content, file.filename, source_region)
    save_json_artifact("groq_symptom", patient_id, groq_res)

    mod_a3 = a3_crosscheck.format_signal(patient_id, groq_res)
    
    if pipeline_state["latest_result"]:
        return {"status": "success", "crosscheck": mod_a3.model_dump(), "latest_result": pipeline_state["latest_result"]}
    
    mock_res = generate_mock_frontend_output(patient_id).model_dump()
    return {"status": "success", "crosscheck": mod_a3.model_dump(), "latest_result": mock_res}

@app.get("/results")
def get_results():
    if pipeline_state["latest_result"]:
        return pipeline_state["latest_result"]
    return generate_mock_frontend_output().model_dump()

# ==================== REMINDER SYSTEM ENDPOINTS ====================

@app.post("/reminders")
def create_reminder_endpoint(payload: ReminderCreate):
    """Create a reminder: sends confirmation email + stores it, then fires any due reminders."""
    if payload.appointment_at <= datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Appointment time must be in the future")
    reminder = create_reminder(payload)
    process_due_reminders()
    return reminder.model_dump()

@app.get("/reminders")
def list_reminders_endpoint(patient_id: str):
    """List reminders for a patient, firing any due lead-time reminder emails first."""
    process_due_reminders()
    reminders = list_reminders(patient_id)
    return [r.model_dump() for r in reminders]

@app.delete("/reminders/{reminder_id}")
def delete_reminder_endpoint(reminder_id: str):
    """Delete a reminder. Returns 404 if it did not exist."""
    if not delete_reminder(reminder_id):
        raise HTTPException(status_code=404, detail="Reminder not found")
    return {"deleted": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.post("/api/reminders/schedule")
def schedule_reminder_alias(payload: ReminderCreate):
    return create_reminder_endpoint(payload)
