"""GraderInputBuilder — real input normalizer for the deployed anemia-workup model.

Consumes the **pdf_raw** extraction artifact (NOT the `biomarkers/` NER folder) and
produces the exact patient-record payload expected by
`vitascan_grader_model.AnemiaGrader.grade(...)`:

    { "hemoglobin": ..., "mcv": ..., "rdw": ..., "ferritin": ...,
      "b12": ..., "folate": ..., "rbc": ..., "mch": ..., "mchc": ... }

Cleaning pipeline per biomarker:
  1. Alias-normalise keys (our slugs -> the model's 9 canonical names).
  2. Coerce to a real number (drop non-numeric GLiNER-style tokens, handle "<value>",
     thousands separators, etc.).
  3. Validate / convert units to the model's training units.
  4. Plausibility gate (reject outliers, warn near-plausible).
  5. If still missing/unresolved, fall back to scanning `raw_text` (removes reference
     range pairs like `83 - 101`, keeps the actual result value).
  6. Missing stays `null` (the model's KNN imputer fills it) — derived indices are
     computed, and (optionally) the deployed KNN imputer is applied for logging.

The builder is self-contained (no torch / grader-model import); it only needs
`numpy` + optional `joblib`/`scikit-learn` for the imputation log.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.path_b_blood_report.mod_b2_normalizer.composer import compose_pdf_raw

logger = logging.getLogger("vitascan.grader_input")

# ---------------------------------------------------------------------------
# Model contract (mirrors vitascan_grader_model/vitascan/features.py)
# ---------------------------------------------------------------------------
MODEL_FEATURE_NAMES = [
    "hemoglobin", "mcv", "rdw", "ferritin", "b12", "folate",
    "rbc", "mch", "mchc", "srivastava_index", "mentzer_index", "cell_hb_density",
]
BASE_NAMES = MODEL_FEATURE_NAMES[:9]

_BC = r"(?<![A-Za-z0-9])"  # boundary: not preceded by alnum
_AC = r"(?![A-Za-z0-9])"   # boundary: not followed by alnum

# Canonical contract: expected units, plausible clinical range, alias keys, text keywords.
MODEL_BIOMARKER_SPEC = {
    "hemoglobin": {
        "unit": "g/dL",
        "plausible": (3.0, 22.0),
        "alias": ["hemoglobin", "hemoglobin_hb", "hb"],
        "text": [r"ha?emoglobin\b(?!\s*\(?\s*(hba1c|a1c))", rf"{_BC}hb{_AC}"],
    },
    "mcv": {
        "unit": "fL",
        "plausible": (55.0, 130.0),
        "alias": ["mean_corpuscular_volume_mcv", "mcv", "mean_corpuscular_volume"],
        "text": [rf"{_BC}mcv{_AC}", r"mean corpuscular volume"],
    },
    "rdw": {
        "unit": "%",
        "plausible": (8.0, 30.0),
        "alias": ["red_cell_distribution_width_rdw", "rdw", "rdw_cv"],
        "text": [rf"{_BC}rdw{_AC}", r"red cell distribution width"],
    },
    "ferritin": {
        "unit": "ng/mL",
        "plausible": (1.0, 3000.0),
        "alias": ["ferritin", "serum_ferritin"],
        "text": [r"ferritin\b", r"serum ferritin"],
    },
    "b12": {
        "unit": "pg/mL",
        "plausible": (20.0, 3000.0),
        "alias": ["b12", "vitamin_b12"],
        "text": [rf"{_BC}b12{_AC}", r"vitamin b[\s\-]?12"],
    },
    "folate": {
        "unit": "ng/mL",
        "plausible": (1.0, 50.0),
        "alias": ["folate", "serum_folate"],
        "text": [r"folate\b", r"serum folate"],
    },
    "rbc": {
        "unit": "million/uL",
        "plausible": (2.0, 8.0),
        "alias": ["total_rbc_count", "rbc_count", "red_blood_cell_count", "rbc"],
        "text": [rf"{_BC}rbc{_AC}", r"total rbc count", r"red blood cell count"],
    },
    "mch": {
        "unit": "pg",
        "plausible": (18.0, 42.0),
        "alias": ["mean_corpuscular_hemoglobin_mch", "mch", "mean_corpuscular_hemoglobin"],
        "text": [rf"{_BC}mch{_AC}", r"mean corpuscular hemoglobin"],
    },
    "mchc": {
        "unit": "g/dL",
        "plausible": (28.0, 40.0),
        "alias": [
            "mean_corpuscular_hemoglobin_concentration_mchc",
            "mchc",
            "mean_corpuscular_hemoglobin_concentration",
        ],
        "text": [rf"{_BC}mchc{_AC}", r"mean corpuscular hemoglobin concentration"],
    },
}

# unit -> multiplicative factor to reach the model's expected unit ("" = identical)
UNIT_FACTORS = {
    "hemoglobin": {"g/dl": 1.0, "gm/dl": 1.0, "g/l": 0.1, "g/l": 0.1},
    "mcv": {"fl": 1.0, "fl/cell": 1.0, "um3": 1.0},
    "rdw": {"%": 1.0, "percent": 1.0},
    "ferritin": {"ng/ml": 1.0, "µg/l": 1.0, "ug/l": 1.0, "mcg/l": 1.0, "ug/ml": 1000.0},
    "b12": {"pg/ml": 1.0, "ng/ml": 1000.0, "pg/l": 0.001, "pmol/l": 1.3554},
    "folate": {"ng/ml": 1.0, "µg/l": 1.0, "ug/l": 1.0, "nmol/l": 0.4413},
    "rbc": {"million/ul": 1.0, "million/cmm": 1.0, "m/ul": 1.0, "x10^6/ul": 1.0,
            "10^6/ul": 1.0, "cells/ul": 1e-6},
    "mch": {"pg": 1.0, "pg/cell": 1.0, "pg/rbc": 1.0},
    "mchc": {"g/dl": 1.0, "gm/dl": 1.0, "%": 1.0},
}

_RANGE_PAIR = re.compile(r"\d+(?:\.\d+)?\s*[-–—]\s*(?:\d+(?:\.\d+)?)")
_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_INEQUALITY = ("<", ">", "<=", ">=", "≤", "≥")


DERIVED_SPEC = [
    ("srivastava_index", "mch / rbc"),
    ("mentzer_index", "mcv / rbc"),
    ("cell_hb_density", "mch * mchc / 100"),
]


def parse_number(raw: Any) -> Optional[float]:
    """Best-effort coercion to float. Returns None when not a real number."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s or s.lower() in {"na", "n/a", "none", "nil", "--", "-", "not done", "negative"}:
        return None
    s = s.replace(",", "")
    match = _NUMBER.search(s)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _unit_factor(biomarker: str, unit: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    """Return (factor, issue) for converting `unit` to the model's expected unit."""
    if not unit or not str(unit).strip():
        return 1.0, None
    key = str(unit).replace("µ", "u").strip().lower()
    factor = UNIT_FACTORS.get(biomarker, {}).get(key)
    if factor is None:
        # a *different* biomarker's unit was attached (e.g. hemoglobin marked "Low")
        return 1.0, f"unit '{unit}' is not a {biomarker} unit"
    return factor, None


class GraderInputBuilder:
    """Turns a `pdf_raw` extraction dict into model-usable patient records."""

    def __init__(self, imputer_joblib: Optional[str] = None):
        self.imputer = None
        imp_path = imputer_joblib or self._default_imputer_path()
        if imp_path and Path(imp_path).exists():
            try:
                import joblib
                self.imputer = joblib.load(imp_path)
            except Exception as e:  # pragma: no cover
                logger.warning("Could not load KNN imputer %s: %s", imp_path, e)

    @staticmethod
    def _default_imputer_path() -> Optional[str]:
        import os
        env = os.environ.get("VITASCAN_GRADER_MODEL_DIR")
        if env:
            return str(Path(env) / "models" / "knn_imputer.joblib")
        here = Path(__file__).resolve()
        for base in (here.parents[3], here.parents[4]):  # vitascan_v3/ or Projects/
            cand = base / "vitascan_grader_model" / "models" / "knn_imputer.joblib"
            if cand.exists():
                return str(cand)
        return None

    # ------------------------------------------------------------------ build
    def build(self, pdf_raw: Dict[str, Any]) -> Dict[str, Any]:
        patient_id = pdf_raw.get("patient_id") or pdf_raw.get("filename") or "PAT-UNKNOWN"
        markers = pdf_raw.get("biomarkers") or {}
        raw_text = pdf_raw.get("raw_text") or ""
        source_file = pdf_raw.get("filename") or pdf_raw.get("json_storage_path")

        resolved: Dict[str, Dict[str, Any]] = {}
        warnings: List[str] = []

        # Stage 1: structured `biomarkers` dict (aliases).
        for name, spec in MODEL_BIOMARKER_SPEC.items():
            found = None
            for alias in spec["alias"]:
                entry = markers.get(alias)
                if not entry or not isinstance(entry, dict):
                    continue
                found = self._clean_entry(name, alias, entry)
                if found["value"] is not None:
                    break
            if not found:
                resolved[name] = {"value": None, "status": "missing", "resolved_from": None}
                warnings.append(f"{name}: not present in pdf_raw biomarkers -> null")
                continue
            resolved[name] = found
            if found["status"] != "ok":
                warnings.append(f"{name}: {found['status']} (value {found['value']}, raw '{found.get('raw_value')!r}')")

        # Stage 2: raw_text fallback for anything still missing.
        if raw_text:
            for name in BASE_NAMES:
                if resolved[name]["value"] is not None:
                    continue
                val = self._scan_text(name, raw_text)
                if val is not None:
                    resolved[name] = {"value": val, "status": "ok", "resolved_from": "raw_text",
                                      "unit": MODEL_BIOMARKER_SPEC[name]["unit"], "raw_value": val}
                else:
                    resolved[name]["resolved_from"] = "raw_text(no-match)"
                    warnings.append(f"{name}: not found in raw_text either -> null")

        model_input = {name: (None if resolved[name]["value"] is None else resolved[name]["value"])
                       for name in BASE_NAMES}

        record = {
            "format_version": "1.0.0",
            "target": "vitascan_grader_model.AnemiaGrader.grade",
            "patient_id": patient_id,
            "source_file": source_file,
            "na_count": sum(1 for v in model_input.values() if v is None),
            "model_input": model_input,
            "biomarkers": {name: resolved[name] for name in BASE_NAMES},
            "derived": self._compute_derived(model_input),
            "warnings": warnings,
        }
        record["imputed"] = self._impute(model_input)
        return record

    def build_composite(
        self,
        docs: List[Dict[str, Any]],
        patient_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compose several per-file pdf_raw dicts, then build the grader record.

        Runs `compose_pdf_raw` (key-level first-file-wins merge + concatenated
        raw_text) and forwards the result through the standard cleaning pipeline.
        The returned record additionally carries a `composite` block with the
        provenance of every part and merged value.
        """
        composite = compose_pdf_raw(docs, patient_id=patient_id)
        record = self.build(composite)
        record["composite"] = {
            "part_count": len(docs),
            "parts": composite["composite_parts"],
            "biomarker_provenance": composite["biomarker_provenance"],
        }
        return record

    # ---------------------------------------------------------------- helpers
    def _clean_entry(self, name: str, alias: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        raw_val = entry.get("value", entry.get("raw_value"))
        unit = entry.get("unit")
        conf = entry.get("confidence")
        out = {
            "value": None, "raw_value": raw_val, "unit": unit, "confidence": conf,
            "resolved_from": alias, "status": "invalid",
        }
        parsed = parse_number(raw_val)
        if parsed is None:
            out["status"] = f"non-numeric value {raw_val!r} -> null"
            return out
        factor, unit_issue = _unit_factor(name, unit)
        value = parsed * factor
        lo, hi = MODEL_BIOMARKER_SPEC[name]["plausible"]
        status = "ok"
        if not (lo <= value <= hi):
            # soft band: outside plausible but within expansive bounds -> warn, keep
            if (lo * 0.5 <= value <= hi * 1.5):
                status = f"value {value} outside plausible [{lo}, {hi}] (kept)"
            else:
                status = f"implausible value {value} outside [{lo}, {hi}] -> null"
                out.update({"value": None, "status": status})
                return out
        if unit_issue:
            status = f"{status} ({unit_issue})" if status != "ok" else f"ok ({unit_issue})"
        out.update({"value": round(value, 4), "status": status})
        return out

    def _compute_derived(self, inp: Dict[str, Any]) -> Dict[str, Optional[float]]:
        mch = inp["mch"]; rbc = inp["rbc"]; mcv = inp["mcv"]; mchc = inp["mchc"]
        eps = 1e-6
        def safe(dividend, divisor):
            if dividend is None or divisor is None:
                return None
            return round(dividend / (divisor + eps), 4)
        return {
            "srivastava_index": safe(mch, rbc),
            "mentzer_index": safe(mcv, rbc),
            "cell_hb_density": None if mch is None or mchc is None else round(mch * mchc / 100.0, 4),
        }

    def _impute(self, inp: Dict[str, Any]) -> Optional[Dict[str, Optional[float]]]:
        """KNN-impute the 12-feature row using the deployed imputer (log only)."""
        if self.imputer is None:
            return None
        try:
            arr = self.imputer.transform(build_feature_matrix([inp]))
            return {name: (None if np.isnan(v) else round(float(v), 4))
                    for name, v in zip(MODEL_FEATURE_NAMES, arr[0])}
        except Exception as e:  # pragma: no cover
            logger.warning("imputation failed: %s", e)
            return None

    # ------------------------------------------------------------ raw_text scan
    def _scan_text(self, name: str, text: str) -> Optional[float]:
        """Find the biomarker's result value in raw text.

        Walks lines after the keyword, building the window incrementally; a value
        is accepted as soon as the window contains a reference-range pair AND
        exactly one clinical-plausible candidate outside any `A - B` pair. This
        handles single-line rows (``Hb (Hb) 12.5 ... 13.0 - 17.0 g/dL``), compact
        columnar blocks (``Hemoglobin\\ng/dL\\n13.0 - 16.5\\n...\\n14.5``) and
        blank-line-separated table cells (``Vitamin B12\\nL\\npg/mL\\n187 - 833\\
        nCLIA\\n< 148``). Prose mentions ("...vitamin B12 or folate...") have no
        adjacent number/range and are skipped.
        """
        lo, hi = MODEL_BIOMARKER_SPEC[name]["plausible"]
        for pattern in MODEL_BIOMARKER_SPEC[name]["text"]:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                res = self._window_viable(text, m.end(), lo, hi)
                if res is not None:
                    return res
        return None

    @staticmethod
    def _window_viable(text: str, pos: int, lo: float, hi: float) -> Optional[float]:
        window = ""
        for i, line in enumerate(text[pos:].split("\n")):
            if i >= 10:
                break
            window = window + "\n" + line if window else line
            viable = GraderInputBuilder._viable_in_window(window, lo, hi)
            if i == 0 and len(viable) == 1 and _NUMBER.search(window):
                return round(viable[0], 4)   # value already on the same line
            if _RANGE_PAIR.search(window) and len(viable) == 1:
                return round(viable[0], 4)
        return None

    @staticmethod
    def _viable_in_window(window: str, lo: float, hi: float) -> List[float]:
        candidates = [float(n.replace(",", "")) for n in _NUMBER.findall(window)]
        ranged = set()
        for rpair in _RANGE_PAIR.finditer(window):
            ranged.update(float(n.replace(",", "")) for n in _NUMBER.findall(rpair.group(0)))
        return [c for c in candidates if c not in ranged and lo <= c <= hi]

    # --------------------------------------------------------------- utilities
    def to_payload(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """The exact dict to feed `AnemiaGrader.grade(...)`."""
        row = {"patient_id": record["patient_id"]}
        for k, v in record["model_input"].items():
            row[k] = v
        return row


def build_feature_matrix(rows) -> np.ndarray:
    """Build the (N, 12) matrix in MODEL_FEATURE_NAMES order (NaN-tolerant).

    Mirrors vitascan.features.build_feature_frame without the pandas dependency:
    9 base values -> +3 derived indices, missing base -> NaN.
    """
    if isinstance(rows, dict):
        rows = [rows]
    matrix = []
    for row in rows:
        missing = [c for c in BASE_NAMES if c not in row]
        if missing:
            raise ValueError(f"Missing base columns: {missing}")
        eps = 1e-6
        def num(key):
            v = row[key]
            if v is None:
                return float("nan")
            return float(v)
        mch, rbc, mcv, mchc = num("mch"), num("rbc"), num("mcv"), num("mchc")
        matrix.append([
            num("hemoglobin"), num("mcv"), num("rdw"), num("ferritin"), num("b12"),
            num("folate"), num("rbc"), num("mch"), num("mchc"),
            mch / (rbc + eps),
            mcv / (rbc + eps),
            (mch * mchc) / 100.0,
        ])
    return np.asarray(matrix, dtype=float)