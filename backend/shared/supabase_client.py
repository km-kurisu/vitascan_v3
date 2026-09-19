"""
VitaScan Supabase Client Integration Layer
Provides thread-safe access to Supabase database operations (patient profiles, scans, and results).
"""
import os
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("vitascan.supabase")

SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "")

_supabase_client = None

def get_supabase_client():
    """Lazily initializes and returns the Supabase Python client."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.info("Supabase credentials not set (SUPABASE_URL/SUPABASE_KEY). Running in local fallback mode.")
        return None

    try:
        from supabase import create_client, Client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Successfully initialized Supabase Client.")
        return _supabase_client
    except Exception as e:
        logger.warning(f"Failed to initialize Supabase client: {e}")
        return None


def save_patient_record(patient_id: str, full_name: Optional[str] = None, age: Optional[int] = None, gender: Optional[str] = None, clerk_user_id: Optional[str] = None) -> bool:
    """Saves or updates a patient record in Supabase."""
    client = get_supabase_client()
    if not client:
        logger.debug(f"[Mock Supabase] Saved patient: {patient_id}")
        return True

    try:
        data = {
            "patient_id": patient_id,
            "clerk_user_id": clerk_user_id,
            "full_name": full_name,
            "age": age,
            "gender": gender,
        }
        res = client.table("patients").upsert(data, on_conflict="patient_id").execute()
        return bool(res.data)
    except Exception as e:
        logger.error(f"Supabase save_patient_record error: {e}")
        return False


def save_patient_biomarkers(patient_id: str, biomarkers_data: Dict[str, Any]) -> bool:
    """Saves BioBERT extracted biomarkers for a patient in database record with logging."""
    logger.info(f"Saving BioBERT patient biomarkers for patient '{patient_id}'...")
    client = get_supabase_client()
    if not client:
        logger.debug(f"[Mock Supabase] Saved BioBERT biomarkers for patient: {patient_id}")
        return True

    try:
        data = {
            "patient_id": patient_id,
            "biomarkers": biomarkers_data.get("biomarkers", {}),
            "model": biomarkers_data.get("model", "biobert-v1.1"),
            "confidence_summary": biomarkers_data.get("confidence_summary", 0.0),
            "updated_at": "now()"
        }
        res = client.table("patient_biomarkers").upsert(data, on_conflict="patient_id").execute()
        logger.info(f"Successfully saved BioBERT patient biomarkers to Supabase database for patient '{patient_id}'.")
        return bool(res.data)
    except Exception as e:
        logger.error(f"Supabase save_patient_biomarkers error for patient '{patient_id}': {e}")
        return False


def save_scan_result(scan_id: str, patient_id: str, mod_c_output: Dict[str, Any], blood_report_url: Optional[str] = None, symptom_photo_url: Optional[str] = None) -> bool:
    """Saves scan result summary and deficiency breakdowns in Supabase."""
    client = get_supabase_client()
    if not client:
        logger.debug(f"[Mock Supabase] Saved scan result: {scan_id} for patient {patient_id}")
        return True

    try:
        summary = mod_c_output.get("summary", {})
        scan_data = {
            "scan_id": scan_id,
            "patient_id": patient_id,
            "blood_report_url": blood_report_url,
            "symptom_photo_url": symptom_photo_url,
            "overall_risk_band": summary.get("overall_risk_band", "unknown"),
            "flagged_count": summary.get("flagged_deficiency_count", 0),
            "mod_c_output": mod_c_output,
        }
        client.table("scans").upsert(scan_data, on_conflict="scan_id").execute()

        # Insert individual deficiencies
        deficiencies = mod_c_output.get("deficiencies", [])
        def_records = []
        for item in deficiencies:
            def_records.append({
                "scan_id": scan_id,
                "deficiency_type": item.get("type", "unknown"),
                "severity_band": item.get("severity", {}).get("band", "none"),
                "score_pct": float(item.get("severity", {}).get("score_pct", "0").replace("%", "") or 0),
                "explanation": item.get("explanation", ""),
                "key_contributors": item.get("key_contributors", []),
                "diet_recommendations": item.get("diet_recommendations", []),
            })

        if def_records:
            client.table("deficiencies").insert(def_records).execute()

        return True
    except Exception as e:
        logger.error(f"Supabase save_scan_result error: {e}")
        return False
