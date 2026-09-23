"""Feature construction for the anemia-workup model.

Matches docs/model.py (training script): 12 features from 9 base CBC/biomarker
values plus 3 derived clinical indices.
"""
from __future__ import annotations

import pandas as pd

# Order matters — must match the trained scaler / GNN feature order.
FEATURE_NAMES = [
    "hemoglobin", "mcv", "rdw", "ferritin", "b12", "folate",
    "rbc", "mch", "mchc", "srivastava_index", "mentzer_index", "cell_hb_density",
]

BASE_NAMES = FEATURE_NAMES[:9]
DEFAULT_EPS = 1e-6


def compute_derived(df: pd.DataFrame, eps: float = DEFAULT_EPS) -> pd.DataFrame:
    """Add the three derived indices exactly as in the training script.

    - srivastava_index  = mch / rbc
    - mentzer_index     = mcv / rbc
    - cell_hb_density   = mch * mchc / 100.0
    """
    out = df.copy()
    out["srivastava_index"] = out["mch"] / (out["rbc"] + eps)
    out["mentzer_index"] = out["mcv"] / (out["rbc"] + eps)
    out["cell_hb_density"] = (out["mch"] * out["mchc"]) / 100.0
    return out


def build_feature_frame(rows) -> pd.DataFrame:
    """Build the 12-column feature DataFrame from records of the 9 base values.

    Accepts a list of dicts, a single dict, or an existing DataFrame (which may
    already contain the derived columns).
    """
    if isinstance(rows, pd.DataFrame):
        df = rows.reset_index(drop=True)
        if all(c in df.columns for c in FEATURE_NAMES):
            return df[FEATURE_NAMES]
        missing = [c for c in BASE_NAMES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing base columns: {missing}")
        return compute_derived(df)[FEATURE_NAMES]

    if isinstance(rows, dict):
        rows = [rows]

    df = pd.DataFrame.from_records(rows)
    missing = [c for c in BASE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing base columns: {missing}")
    return compute_derived(df)[FEATURE_NAMES]