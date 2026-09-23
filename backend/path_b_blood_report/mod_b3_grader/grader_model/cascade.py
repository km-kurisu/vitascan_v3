"""XGBoost 3-stage hierarchical cascade — replicate of HierarchicalXGBoostCascade.

The deployed bundle stores the three fitted estimators in a dict
(``xgb_cascade.joblib``). We wrap them back into the cascade and reproduce the
exact joint 4-class probability assembly from docs/model.py.
"""
from __future__ import annotations

import joblib
from pathlib import Path

import numpy as np

ETIOLOGY_NAMES = ["No Anemia", "IDA", "B12 Deficiency", "Folate Deficiency"]


class XGBCascadePredictor:
    """Loaded-stages version of the training-time HierarchicalXGBoostCascade."""

    def __init__(self, model_dir: Path):
        bundle: dict = joblib.load(model_dir / "xgb_cascade.joblib")
        self.stage1_xgb = bundle["stage1_xgb"]            # 0 healthy vs 1 anemia
        self.stage2a_xgb = bundle["stage2a_xgb"]          # 0 IDA/B12 vs 1 folate
        self.stage2b_calibrated = bundle["stage2b_calibrated"]  # 0 IDA vs 1 B12 (isotonic)

    def predict_proba(self, X) -> np.ndarray:
        """Assemble the joint 4-class distribution (exact replica of training code)."""
        X = np.asarray(X, dtype=float)
        n = len(X)
        probs = np.zeros((n, 4))

        p_anemia = self.stage1_xgb.predict_proba(X)[:, 1]          # P(any anemia)
        p_healthy = 1.0 - p_anemia

        p_folate_given_anemia = self.stage2a_xgb.predict_proba(X)[:, 1]
        p_diff_given_anemia = 1.0 - p_folate_given_anemia          # fol=0 -> IDA|B12

        p_b12_given_diff = self.stage2b_calibrated.predict_proba(X)[:, 1]
        p_ida_given_diff = 1.0 - p_b12_given_diff

        probs[:, 0] = p_healthy
        probs[:, 1] = p_anemia * p_diff_given_anemia * p_ida_given_diff
        probs[:, 2] = p_anemia * p_diff_given_anemia * p_b12_given_diff
        probs[:, 3] = p_anemia * p_folate_given_anemia
        return probs