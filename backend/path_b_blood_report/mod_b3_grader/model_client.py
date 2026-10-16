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
from typing import Any, Dict, Optional, Set

from backend.path_b_blood_report.mod_b3_grader.explainability import (
    counterfactual_flips,
    explain_patient,
)
from backend.path_b_blood_report.mod_b3_grader.explainability.attributions import (
    DERIVED_PARENTS,
    FEATURE_LABELS,
)
from backend.path_b_blood_report.mod_b3_grader.grader_model.features import BASE_NAMES
from backend.path_b_blood_report.mod_b3_grader.grader_model.predict import (
    AnemiaGrader,
    DECISION_NAMES,
)
from backend.shared.schemas import (
    CounterfactualExplanation,
    CounterfactualTarget,
    FeatureAttribution,
    ModelExplanation,
    ModelGradeDetail,
)

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

#: Counterfactual search costs seconds per patient -- the GNN cannot be batched
#: (see ``explainability.counterfactual``), so each boundary is a grid sweep
#: plus a bisection of single-row forward passes. Off by default so the upload
#: path stays fast; switch it on where the distance to the boundary is worth
#: the wait, which is mainly the grey zone.
COUNTERFACTUALS_DEFAULT = False


def get_grader(model_dir: Optional[Any] = None) -> AnemiaGrader:
    """Lazily construct the singleton AnemiaGrader (heavy torch/pyg import)."""
    global _GRADER
    if _GRADER is None:
        _GRADER = AnemiaGrader(model_dir) if model_dir else AnemiaGrader()
    return _GRADER


def model_feature_names():
    return None


def imputed_feature_names(record: Dict[str, Any]) -> Set[str]:
    """Which of the 12 model features the KNN imputer had to supply.

    ``record["imputed"]`` is the *post-imputation* 12-feature row, not a record
    of what was guessed -- every entry is populated, so it cannot answer this.
    Provenance comes from ``model_input`` instead: a missing base value is one
    the imputer filled. A derived index counts as imputed when any value it is
    computed from was missing, since it is rebuilt from imputed parents.
    """
    model_input = record.get("model_input") or {}
    missing_base = {name for name in BASE_NAMES if model_input.get(name) is None}
    imputed = set(missing_base)
    for derived, parents in DERIVED_PARENTS.items():
        if any(parent in missing_base for parent in parents):
            imputed.add(derived)
    return imputed


def _build_explanation(
    grader: AnemiaGrader, row: Dict[str, Any], record: Dict[str, Any]
) -> Optional[ModelExplanation]:
    """Real per-feature attribution for this verdict, or ``None``.

    Deliberately swallows its own failures. An explanation is a reporting
    feature; a patient's verdict must never depend on SHAP being installed, on a
    tree being explainable, or on a torch version agreeing with us. Anything
    that goes wrong here degrades to "no explanation", which the formatter
    already handles -- but it is logged loudly, because a silent fallback in a
    medical tool is how a fabricated number ends up back on screen.
    """
    try:
        explanation = explain_patient(
            grader,
            row,
            imputed_features=imputed_feature_names(record),
        )
    except Exception:
        logger.exception(
            "Attribution failed for %s; verdict stands, explanation omitted",
            record.get("patient_id"),
        )
        return None

    ranked = sorted(
        explanation.contributions, key=lambda c: abs(c.share_blended), reverse=True
    )
    return ModelExplanation(
        explained_class=explanation.explained_class,
        explained_label=explanation.explained_label,
        contributions=[
            FeatureAttribution(
                feature=c.feature,
                label=c.label,
                share_pct=round(c.share_blended * 100, 1),
                estimated=c.was_imputed,
                is_derived=c.is_derived,
                derived_from=list(c.derived_from),
            )
            for c in ranked
        ],
        p_b12_conditional=explanation.p_b12_conditional,
        grey_zone_band=(
            list(explanation.grey_zone_band) if explanation.grey_zone_band else None
        ),
        gnn_available=explanation.gnn_available,
        caveats=list(explanation.caveats),
    )


def _build_counterfactuals(
    grader: AnemiaGrader, row: Dict[str, Any], record: Dict[str, Any]
) -> Optional[CounterfactualExplanation]:
    """What would move this verdict. Swallows its own failures, like attribution.

    The verdict must never depend on this succeeding. It is also the slowest
    thing in this module by an order of magnitude, which is why it is off by
    default rather than merely optional.
    """
    try:
        report = counterfactual_flips(
            grader,
            row,
            imputed_features=imputed_feature_names(record),
        )
    except Exception:
        logger.exception(
            "Counterfactual search failed for %s; verdict stands, omitted",
            record.get("patient_id"),
        )
        return None

    def _label(name: str) -> str:
        return FEATURE_LABELS.get(name, name)

    return CounterfactualExplanation(
        targets=[
            CounterfactualTarget(
                feature=flip.feature,
                label=flip.label,
                unit=flip.unit,
                # Two decimals: the bisection tolerance is ~1e-3 of each
                # marker's searched span, so anything past the second decimal
                # would be false precision dressed up as a lab value.
                current_value=round(flip.current_value, 2),
                target_value=round(flip.target_value, 2),
                direction=flip.direction,
                to_etiology=flip.to_etiology,
                at_range_boundary=flip.at_range_boundary,
            )
            for flip in report.flips
        ],
        unreachable=[_label(name) for name in report.unreachable],
        refused=[_label(name) for name in report.refused],
        caveats=list(report.caveats),
    )


def grade_record(
    record: Dict[str, Any],
    explain: bool = True,
    counterfactuals: bool = COUNTERFACTUALS_DEFAULT,
) -> ModelGradeDetail:
    """Run the deployed model on one GraderInputBuilder record.

    ``explain`` controls whether real attributions are attached. Explainability
    costs a SHAP pass per cascade stage plus an Integrated Gradients sweep, so
    it is switchable; the verdict itself is identical either way.

    ``counterfactuals`` additionally runs the boundary search, which costs
    seconds rather than milliseconds. Both flags change only what is reported.
    """
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
        explanation=_build_explanation(grader, row, record) if explain else None,
        counterfactuals=(
            _build_counterfactuals(grader, row, record) if counterfactuals else None
        ),
    )


def severity_overlay(decision_code: int) -> Dict[str, tuple]:
    """Return the severity overlay for a model decision (deficiency -> band, score)."""
    return MODEL_DECISION_TO_SEVERITY.get(int(decision_code), {})


def decision_etiology(decision_code: int) -> str:
    return DECISION_NAMES[int(decision_code)]