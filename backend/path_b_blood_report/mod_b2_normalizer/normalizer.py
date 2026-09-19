import json
import logging
import re
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("vitascan.normalizer")

def clean_float(val_raw: Any) -> float:
    if isinstance(val_raw, (int, float)):
        return float(val_raw)
    if not val_raw:
        return 0.0
    val_str = str(val_raw).strip()
    match = re.search(r"[-+]?\d*\.\d+|\d+", val_str)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return 0.0
    return 0.0

class BiomarkerNormalizer:
    def __init__(self):
        ref_path = Path(__file__).parent / "biomarker_reference.json"
        with open(ref_path, "r", encoding="utf-8") as f:
            self.ref_db = json.load(f)

    def normalize(self, raw_biomarkers: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}
        for key, info in raw_biomarkers.items():
            if isinstance(info, dict):
                raw_val = info.get("value", 0)
                unit_str = info.get("unit", "")
            else:
                raw_val = info
                unit_str = ""

            val = clean_float(raw_val)
            ref = self.ref_db.get(key.lower(), {"min": 0.1, "max": 100.0, "unit": ""})
            ref_min = ref["min"]
            ref_max = ref["max"]

            deviation = 0.0
            if val < ref_min:
                deviation = -1.0 * (ref_min - val) / ref_min
            elif val > ref_max:
                deviation = (val - ref_max) / ref_max

            normalized[key] = {
                "raw_value": val,
                "unit": unit_str or ref.get("unit", ""),
                "ref_min": ref_min,
                "ref_max": ref_max,
                "deviation": round(deviation, 3),
                "is_abnormal": deviation != 0.0
            }
        return normalized
