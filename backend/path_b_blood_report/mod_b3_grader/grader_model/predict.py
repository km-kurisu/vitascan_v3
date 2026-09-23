"""End-to-end inference for the anemia-workup hybrid model.

Pipeline (exact match to docs/model.py + saved deployment bundle):
  raw 9-feature CBC  ->  12 features (3 derived indices)
  -> KNNImputer      ->  StandardScaler
  -> XGB cascade 4-class proba   (stage1 -> stage2a -> stage2b)
  -> Hetero-GNN 4-class proba    (SAGEConv x2, aggr=mean)
  -> soft-voting blend p = alpha*p_xgb + (1-alpha)*p_gnn
  -> decision triage cascade -> {0 No Anemia, 1 IDA, 2 B12, 3 Folate, 4 Grey Zone}
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .cascade import XGBCascadePredictor, ETIOLOGY_NAMES
from .features import FEATURE_NAMES, build_feature_frame
from .gnn import load_gnn, build_hetero_clinical_graph

DECISION_NAMES = ETIOLOGY_NAMES + ["Grey Zone Triage"]

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class AnemiaGrader:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = Path(model_dir)

        self.config = joblib.load(self.model_dir / "pipeline_config.joblib")
        self.scaler = joblib.load(self.model_dir / "standard_scaler.joblib")
        self.imputer = joblib.load(self.model_dir / "knn_imputer.joblib")
        self.cascade = XGBCascadePredictor(self.model_dir)
        self.gnn, self.metadata = load_gnn(self.model_dir)

        self.alpha = float(self.config["alpha"])
        self.lower_bound = float(self.config.get("lower_bound", 0.25))
        self.upper_bound = float(self.config.get("upper_bound", 0.75))

    # ------------------------------------------------------------------ pre
    def preprocess(self, rows) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (imputed, scaled) 12-column DataFrames."""
        df = build_feature_frame(rows)[FEATURE_NAMES]
        imputed = pd.DataFrame(
            self.imputer.transform(df.to_numpy(dtype=float)),
            columns=FEATURE_NAMES,
        )
        scaled = pd.DataFrame(
            self.scaler.transform(imputed.to_numpy(dtype=float)),
            columns=FEATURE_NAMES,
        )
        return imputed, scaled

    # ------------------------------------------------------------------ gnn
    @torch.inference_mode()
    def predict_gnn(self, scaled_df) -> np.ndarray:
        graph = build_hetero_clinical_graph(scaled_df)
        out = self.gnn(graph.x_dict, graph.edge_index_dict)
        return F.softmax(out, dim=1).cpu().numpy()

    # ------------------------------------------------------------------ full
    def predict_proba(self, rows) -> dict:
        imputed, scaled = self.preprocess(rows)
        X_imp = imputed.to_numpy(dtype=float)   # cascade trained on IMPUTED features
        p_xgb = self.cascade.predict_proba(X_imp)
        p_gnn = self.predict_gnn(scaled)        # GNN trained on SCALED features
        p_blend = (self.alpha * p_xgb) + ((1.0 - self.alpha) * p_gnn)

        decisions = []
        for p in p_blend:
            p_b12_cond = (p[2] / (p[1] + p[2])) if (p[1] + p[2]) > 0 else 0.5
            if p[0] >= 0.50:
                decisions.append(0)
            elif p[3] >= 0.40:
                decisions.append(3)
            elif self.lower_bound <= p_b12_cond <= self.upper_bound:
                decisions.append(4)
            elif p_b12_cond > self.upper_bound:
                decisions.append(2)
            else:
                decisions.append(1)

        return {
            "proba_blended": p_blend,
            "proba_xgb": p_xgb,
            "proba_gnn": p_gnn,
            "decision": np.asarray(decisions, dtype=int),
            "decision_names": DECISION_NAMES,
        }

    # ------------------------------------------------------------- pretty out
    def grade(self, rows) -> list[dict]:
        """Public API — returns per-patient record dicts (JSON-serialisable)."""
        if isinstance(rows, dict):
            rows = [rows]
        imputed, scaled = self.preprocess(rows)
        probs = self.predict_proba(rows)

        records = []
        for i in range(len(rows)):
            p = probs["proba_blended"][i]
            d = int(probs["decision"][i])
            pid = rows[i].get("patient_id", f"PAT-{i:03d}") if isinstance(rows[i], dict) else f"PAT-{i:03d}"
            records.append({
                "patient": pid,
                "etiology": DECISION_NAMES[d],
                "decision_code": d,
                "proba": {name: round(float(v), 4) for name, v in zip(ETIOLOGY_NAMES, p)},
                "proba_xgb": {name: round(float(v), 4) for name, v in zip(ETIOLOGY_NAMES, probs["proba_xgb"][i])},
                "proba_gnn": {name: round(float(v), 4) for name, v in zip(ETIOLOGY_NAMES, probs["proba_gnn"][i])},
                "alpha": self.alpha,
            })
        return records


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Run the deployed anemia-workup hybrid model")
    ap.add_argument("--patients", required=True, help="JSON file or inline JSON with a list of patient records")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    args = ap.parse_args()

    grader = AnemiaGrader(args.model_dir)
    raw = json.loads(args.patients)
    rows = raw if isinstance(raw, list) else [raw]
    print(json.dumps(grader.grade(rows), indent=2))


if __name__ == "__main__":
    main()