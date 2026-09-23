"""AnemiaGraderClient — adapter between the deployed model (copied from
`vitascan_grader_model` into `mod_b3_grader/grader_model`) and the Vitascan V3
mod-b3 flow.

Input:  a `GraderInputBuilder.build(...)` record (9-key `model_input`, missing
        biomarkers as `None`).
Output: a `ModelGradeDetail` pydantic block embedded in `ModB3Output.model`, so
        the ModC formatter and the REST responses carry the hybrid model verdict
        (xgb cascade + hetero-GNN blend) alongside the legacy stage severity.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from backend.path_b_blood_report.mod_b3_grader.grader_model.predict import (
    AnemiaGrader,
    DECISION_NAMES,
)
from backend.shared.schemas import ModelGradeDetail

logger = logging.getLogger("vitascan.mod_b3.model")

# Model decision -> {deficiency -> (severity band, score)} overlay used to keep
# ModB3Output.severity in sync with the deployed model's verdict.
MODEL_DECISION_TO_SEVERITY: Dict[int, Dict[str, tuple]] = {
    0: {},  # No Anemia -> nothing flagged
    1: {"iron": ("moderate", 0.72), "anemia": ("moderate", 0.75)},  # IDA
    2: {"b12": ("moderate", 0.72)},  # B12 deficiency
    3: {"folate": ("moderate", 0.72)},  # Folate deficiency
    4: {"anemia": ("mild", 0.55)},  # Grey Zone Triage -> mild, needs review
}

_GRADER: Optional[AnemiaGrader] = None


def get_grader(model_dir: Optional[Any] = None) -> AnemiaGrader:
    """Lazily construct the singleton AnemiaGrader (heavy torch/pyg import)."""
    global _GRADER
    if _GRADER is None:
        _GRADER = AnemiaGrader(model_dir) if model_dir else AnemiaGrader()
    return _GRADER


def model_feature_names():
    return None


def grade_record(record: Dict[str, Any]) -> ModelGradeDetail:
    """Run the deployed model on one GraderInputBuilder record."""
    grader = get_grader()
    row = {"patient_id": record["patient_id"], **record["model_input"]}
    result = grader.grade(row)[0]

    imputed = None
    if record.get("imputed"):
        imputed = {k: (None if v is None else float(v)) for k, v in record["imputed"].items()}

    return ModelGradeDetail(
        decision_code=result["decision_code"],
        etiology=result["etiology"],
        na_count=record["na_count"],
        complete_input=record["na_count"] == 0,
        proba=result["proba"],
        proba_xgb=result["proba_xgb"],
        proba_gnn=result["proba_gnn"],
        alpha=result["alpha"],
        imputed_features=imputed,
    )


def severity_overlay(decision_code: int) -> Dict[str, tuple]:
    """Return the severity overlay for a model decision (deficiency -> band, score)."""
    return MODEL_DECISION_TO_SEVERITY.get(int(decision_code), {})


def decision_etiology(decision_code: int) -> str:
    return DECISION_NAMES[int(decision_code)]