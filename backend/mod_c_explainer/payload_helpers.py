"""Presentation helpers for the Mod C payload.

Builds the pieces the results screen needs but that are not part of the
Mod B3 -> Mod C contract: the key blood parameters table and the patient
metadata read off the uploaded report (instead of hardcoded placeholders).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.shared.schemas import BloodParameter, PatientInfo

# Same reference DB the normalizer scores against, so a row's status always
# matches the interval the normalizer used.
_REF_DB_PATH = (
    Path(__file__).resolve().parents[1]
    / "path_b_blood_report/mod_b2_normalizer/biomarker_reference.json"
)
_REFERENCE_DB: Dict[str, Dict[str, Any]] = (
    json.loads(_REF_DB_PATH.read_text(encoding="utf-8")) if _REF_DB_PATH.is_file() else {}
)

# Clinically meaningful markers in display order. Everything else the extractor
# picks up (dates, "Values", free text) is dropped: this table is "key blood
# parameters", not a dump of the NER output.
KEY_PARAMS: Dict[str, str] = {
    "hemoglobin": "Hemoglobin",
    "rbc": "Total RBC Count",
    "pcv": "Packed Cell Volume (PCV)",
    "hematocrit": "Hematocrit",
    "mcv": "Mean Corpuscular Volume (MCV)",
    "mch": "Mean Corpuscular Hemoglobin (MCH)",
    "mchc": "Mean Corpuscular Hb Concentration (MCHC)",
    "rdw": "RDW",
    "wbc": "Total Leucocyte Count (TLC)",
    "platelet_count": "Platelet Count",
    "ferritin": "Serum Ferritin",
    "iron": "Serum Iron",
    "tibc": "Total Iron Binding Capacity (TIBC)",
    "transferrin_saturation": "Transferrin Saturation",
    "b12": "Vitamin B12",
    "folate": "Serum Folate",
    "vitamin_d": "Vitamin D",
    "calcium": "Serum Calcium",
    "zinc": "Serum Zinc",
    "albumin": "Serum Albumin",
}

# NER label variants that mean one of the keys above.
ALIASES: Dict[str, str] = {
    "hgb": "hemoglobin",
    "hb": "hemoglobin",
    "haemoglobin": "hemoglobin",
    "vitamin_b12": "b12",
    "b_12": "b12",
    "cyanocobalamin": "b12",
    "serum_b12": "b12",
    "folic_acid": "folate",
    "serum_folate": "folate",
    "serum_ferritin": "ferritin",
    "serum_iron": "iron",
    "mean_corpuscular_volume": "mcv",
    "mean_corpuscular_hemoglobin": "mch",
    "mean_corpuscular_hb_concentration": "mchc",
    "red_cell_distribution_width": "rdw",
    "packed_cell_volume": "pcv",
    "platelets": "platelet_count",
    "platelet": "platelet_count",
    "wbc_count": "wbc",
    "leucocyte_count": "wbc",
    "total_rbc_count": "rbc",
    "25_oh_vitamin_d": "vitamin_d",
    "vitamin_d_25": "vitamin_d",
    "serum_calcium": "calcium",
    "serum_zinc": "zinc",
    "serum_albumin": "albumin",
}

# The extractor labels markers as "<name>_<abbreviation>" (mean_corpuscular_volume_mcv,
# total_iron_binding_capacity_tibc, vitamin_b12_cyanocobalamin). Suffix rules run
# before the substring fallback so "mean_corpuscular_hemoglobin_mch" lands on mch
# rather than on the "hemoglobin" it also contains.
SUFFIXES: Dict[str, str] = {
    "_mchc": "mchc",
    "_cyanocobalamin": "b12",
    "_tibc": "tibc",
    "_tlc": "wbc",
    "_mcv": "mcv",
    "_mch": "mch",
    "_pcv": "pcv",
    "_hb": "hemoglobin",
    "_cv": "rdw",
    "_rbc": "rbc",
    "_wbc": "wbc",
}

# The 12 features the anemia grader consumes (after KNN imputation), used to
# badge the rows the verdict actually depends on.
MODEL_INPUT_KEYS = {
    "hemoglobin",
    "mcv",
    "rdw",
    "ferritin",
    "b12",
    "folate",
    "rbc",
    "mch",
    "mchc",
    "srivastava_index",
    "mentzer_index",
    "cell_hb_density",
}

# Inside this fraction of the nearest limit a value is "borderline", not low/high.
BORDERLINE_PCT = 0.05

_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)")


def _canonical(key: str) -> str:
    k = str(key).strip().lower()
    if k in KEY_PARAMS:
        return k
    if k in ALIASES:
        return ALIASES[k]
    for suffix, canonical in SUFFIXES.items():
        if k.endswith(suffix) and len(k) > len(suffix) + 2:
            return canonical
    for known in sorted(KEY_PARAMS, key=len, reverse=True):
        if k.startswith(known) or known in k.split("_"):
            return known
    return k


def _display_name(canonical: str, entry: Dict[str, Any]) -> str:
    if canonical in KEY_PARAMS:
        return KEY_PARAMS[canonical]
    ref = _REFERENCE_DB.get(canonical) or {}
    name = str(ref.get("name") or entry.get("canonical_name") or "").strip()
    return name or canonical.replace("_", " ").title()


def _bounds(
    canonical: str, entry: Dict[str, Any], normalized: Dict[str, Any]
) -> Tuple[Optional[float], Optional[float], str]:
    """Reference interval to display *and* to judge status against.

    The interval the lab printed wins; otherwise the reference DB the
    normalizer scores against. Anything else is unusable — the normalizer's
    0.1-100 placeholder must never surface as a reference interval.
    """
    printed_range = _parsed_range(entry)
    if printed_range:
        return printed_range[0], printed_range[1], f"{printed_range[0]:g} - {printed_range[1]:g}"

    if canonical in _REFERENCE_DB:
        ref = _REFERENCE_DB[canonical]
        low, high = float(ref["min"]), float(ref["max"])
        if high > low:
            return low, high, f"{low:g} - {high:g}"

    return None, None, "N/A"


def _parsed_range(entry: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    printed = str(entry.get("ref_range") or "").strip()
    if not printed or printed.upper() == "N/A":
        return None
    match = _RANGE_RE.search(printed)
    if not match:
        return None
    low, high = float(match.group(1)), float(match.group(2))
    return (low, high) if high > low else None


def _scale_for_units(value: float, unit: str, high: Optional[float]) -> Tuple[float, str]:
    """Indian lab convention: platelet/TLC counts printed in lakh or thousand
    per mm3 against a reference interval in absolute counts."""
    if high is None or high <= 0:
        return value, unit
    u = unit.lower().replace(" ", "")
    if "lac" in u and high > 10_000 and value < high / 1000:
        return value * 100_000, "/uL"
    if "thousand" in u and high > 1_000 and value < high / 1000:
        return value * 1_000, "/uL"
    return value, unit


def _deviation(value: float, low: Optional[float], high: Optional[float]) -> float:
    if low is None or high is None:
        return 0.0
    if value < low and low > 0:
        return round(-(low - value) / low, 3)
    if high > 0 and value > high:
        return round((value - high) / high, 3)
    return 0.0


def _status(value: float, low: Optional[float], high: Optional[float]) -> str:
    """Out of range reads Low/High; inside the interval but hugging a limit
    reads Borderline (the lab's own 'watch this' zone)."""
    if low is None or high is None or high <= low:
        return "Normal"
    if value < low:
        return "Low"
    if value > high:
        return "High"
    if low > 0 and value <= low * (1 + BORDERLINE_PCT):
        return "Borderline"
    if high > 0 and value >= high * (1 - BORDERLINE_PCT):
        return "Borderline"
    return "Normal"


# Text patterns used to recover a reference interval for a marker the NER
# missed, by scanning the raw report lines.
TEXT_HINTS: Dict[str, Tuple[str, ...]] = {
    "folate": ("folate", "folic acid"),
    "b12": ("vitamin b12", "b12", "cyanocobalamin"),
    "vitamin_d": ("vitamin d", "25-hydroxyvitamin d", "cholecalciferol"),
    "ferritin": ("ferritin",),
    "iron": ("serum iron", "iron, serum"),
    "tibc": ("tibc", "iron binding capacity", "transferrin"),
    "hemoglobin": ("hemoglobin", " hb"),
    "mcv": ("mcv", "mean corpuscular volume"),
    "mch": ("mch", "mean corpuscular hemoglobin"),
    "mchc": ("mchc", "mean corpuscular hb concentration"),
    "rbc": ("rbc", "red cell count"),
    "pcv": ("pcv", "packed cell volume", "hematocrit", "haematocrit"),
    "wbc": ("wbc", "leucocyte", "leukocyte"),
    "platelet_count": ("platelet",),
    "calcium": ("calcium",),
    "zinc": ("zinc",),
    "albumin": ("albumin",),
}

_PAREN_RANGE_RE = re.compile(r"\(?\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*\)?")


def _range_from_text(raw_text: str, canonical: str) -> Optional[Tuple[float, float]]:
    """Find the interval the report printed for a marker, if any line has it."""
    hints = TEXT_HINTS.get(canonical)
    if not hints or not raw_text:
        return None
    for line in raw_text.splitlines():
        low_line = line.lower()
        if not any(hint in low_line for hint in hints):
            continue
        for match in _PAREN_RANGE_RE.finditer(line):
            low, high = float(match.group(1)), float(match.group(2))
            if 0 < low < high < 1_000_000:
                return low, high
    return None


def _quality(row: BloodParameter) -> tuple:
    """Rank competing readings of the same marker."""
    in_range = row.status in ("Low", "High")
    return (1 if in_range else 0, bool(row.unit), row.confidence)


def build_blood_parameters(
    extraction: Optional[Dict[str, Any]] = None,
    normalized: Optional[Dict[str, Any]] = None,
    model_input: Optional[Dict[str, Any]] = None,
) -> List[BloodParameter]:
    """Turn extractor + normalizer output into the Key Blood Parameters rows.

    `normalized` supplies the numeric value the normalizer scored;
    `extraction` supplies the printed range, unit and extractor confidence.
    Markers the NER missed are backfilled from `model_input` so the table
    still shows what the anemia grader actually scored.
    """
    extraction = extraction or {}
    normalized = normalized or {}
    raw_text = str(extraction.get("raw_text") or "")
    entries = extraction.get("biomarkers") or {}
    if not isinstance(entries, dict):
        entries = {}

    rows: Dict[str, BloodParameter] = {}
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        canonical = _canonical(key)
        if canonical not in KEY_PARAMS and canonical not in _REFERENCE_DB:
            continue

        norm = normalized.get(key) or normalized.get(canonical) or {}
        try:
            value = float(norm.get("raw_value", entry.get("value")))
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue

        unit = str(entry.get("unit") or norm.get("unit") or _REFERENCE_DB.get(canonical, {}).get("unit") or "").strip()
        low, high, range_text = _bounds(canonical, entry, norm)
        if low is None:
            continue
        value, unit = _scale_for_units(value, unit, high)

        try:
            confidence = float(entry.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        row = BloodParameter(
            key=canonical,
            name=_display_name(canonical, entry),
            value=round(value, 2),
            unit=unit,
            normal_range=range_text,
            status=_status(value, low, high),
            deviation=_deviation(value, low, high),
            model_input=canonical in MODEL_INPUT_KEYS,
            confidence=round(min(max(confidence, 0.0), 1.0), 2),
        )

        # The same marker often arrives twice (e.g. "platelets" and
        # "platelet_count"). Keep the better-evidenced reading.
        previous = rows.get(canonical)
        if previous is None or _quality(row) > _quality(previous):
            rows[canonical] = row

    # Backfill: the grader's own input row covers markers the NER dropped
    # (IND-05's serum folate, for one), so the table cannot disagree with the
    # verdict about which values were available.
    for key, raw_value in (model_input or {}).items():
        canonical = _canonical(key)
        if canonical in rows or canonical not in KEY_PARAMS:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue

        bounds = _range_from_text(raw_text, canonical)
        if bounds is None and canonical in _REFERENCE_DB:
            ref = _REFERENCE_DB[canonical]
            bounds = (float(ref["min"]), float(ref["max"]))
        if bounds is None:
            continue
        low, high = bounds
        rows[canonical] = BloodParameter(
            key=canonical,
            name=_display_name(canonical, {}),
            value=round(value, 2),
            unit=str(_REFERENCE_DB.get(canonical, {}).get("unit") or ""),
            normal_range=f"{low:g} - {high:g}",
            status=_status(value, low, high),
            deviation=_deviation(value, low, high),
            model_input=canonical in MODEL_INPUT_KEYS,
            confidence=0.0,
        )

    order = {k: i for i, k in enumerate(KEY_PARAMS)}
    return sorted(rows.values(), key=lambda r: (order.get(r.key, len(order)), r.name))


def resolve_patient(
    extraction: Optional[Dict[str, Any]] = None,
    patient_id: str = "PAT-DEMO123",
) -> PatientInfo:
    """Read age/gender off the report instead of inventing them."""
    extraction = extraction or {}
    details = extraction.get("patient_details") or {}
    labs = extraction.get("labs") or {}
    if not isinstance(labs, dict):
        labs = {}
    raw_text = str(extraction.get("raw_text") or "")

    def _lookup(sources: Tuple[Dict[str, Any], ...], keys: Tuple[str, ...]) -> str:
        """First non-empty value for these keys. Lab values arrive as plain
        strings or as {"Value": ..., "Unit": ...} dicts."""
        for source in sources:
            if not isinstance(source, dict):
                continue
            for key in keys:
                raw = source.get(key)
                if isinstance(raw, dict):
                    raw = raw.get("Value", raw.get("value"))
                if raw not in (None, ""):
                    return str(raw).strip()
        return ""

    age_raw = _lookup((details, extraction, labs), ("age", "Age"))
    age_digits: Optional[str] = None
    inline = re.search(r"\d{1,3}", age_raw)
    if inline:
        age_digits = inline.group(0)
    else:
        labelled = re.search(
            r"Age\s*/?\s*Gender\s*[:\-]?\s*(\d{1,3})", raw_text, re.IGNORECASE
        )
        age_digits = labelled.group(1) if labelled else None
    try:
        age = int(age_digits) if age_digits else None
    except (TypeError, ValueError):
        age = None
    if age is not None and not 0 < age < 130:
        age = None

    gender_raw = _lookup((details, extraction, labs), ("gender", "Gender", "sex", "Sex"))
    if gender_raw.lower() in ("", "n/a", "na", "none", "unknown", "-", "not specified"):
        match = re.search(
            r"Age\s*/?\s*Gender\s*[:\-]?\s*[0-9]{1,3}\s*[/, ]\s*([A-Za-z]+)",
            raw_text,
            re.IGNORECASE,
        )
        gender_raw = match.group(1) if match else ""
    gender = gender_raw.title() if gender_raw else "Unknown"

    return PatientInfo(
        patient_id=str(extraction.get("patient_id") or patient_id),
        age=age,
        gender=gender or "Unknown",
    )
