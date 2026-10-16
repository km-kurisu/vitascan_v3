"""Per-patient explanation assembly (Level 1 orchestrator).

Ties the two halves together -- exact per-stage SHAP for the XGBoost cascade and
Integrated Gradients for the hetero-GNN -- into the per-feature picture the
forecast layer consumes.

On units, and why the blend is on shares
----------------------------------------
The model's headline blend is valid for probabilities::

    p = alpha * p_xgb + (1 - alpha) * p_gnn

It does **not** follow that the explanations blend the same way. TreeSHAP yields
attributions in log-probability units; Integrated Gradients yields them in
pre-softmax logit units. Adding those together would be adding quantities with
different dimensions, and multiplying by ``alpha`` would not fix that.

So this module converts each branch to *shares* -- each feature's fraction of
the total absolute contribution within its own branch -- and blends the shares.
That is a deliberate trade: the shares are dimensionless and therefore
commensurable, but the blend is no longer the model's own ``alpha`` blend
applied exactly. ``caveats`` says so on every result, so nothing downstream can
present it as more exact than it is.

The decision itself is never recomputed here. It is read from
``AnemiaGrader.predict_proba``, so the explanation can never disagree with the
verdict about which verdict it is explaining.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from ..grader_model.features import BASE_NAMES, FEATURE_NAMES
from .gnn_attribution import DEFAULT_STEPS, explain_gnn
from .stage_attribution import (
    ATTRIBUTION_CLASSES,
    compose_class_attribution,
    explain_xgb,
)

DECISION_NAMES = ATTRIBUTION_CLASSES + ["Grey Zone Triage"]

#: Human labels for the 12 model features.
FEATURE_LABELS: Dict[str, str] = {
    "hemoglobin": "Hemoglobin",
    "mcv": "MCV",
    "rdw": "RDW",
    "ferritin": "Ferritin",
    "b12": "Vitamin B12",
    "folate": "Folate",
    "rbc": "RBC count",
    "mch": "MCH",
    "mchc": "MCHC",
    "srivastava_index": "Srivastava index",
    "mentzer_index": "Mentzer index",
    "cell_hb_density": "Cell hemoglobin density",
}

#: The three derived indices, and the measured values they are computed from.
DERIVED_PARENTS: Dict[str, Sequence[str]] = {
    "srivastava_index": ("mch", "rbc"),
    "mentzer_index": ("mcv", "rbc"),
    "cell_hb_density": ("mch", "mchc"),
}


@dataclass
class FeatureContribution:
    """One feature's share of the explanation for a single class."""

    feature: str
    label: str
    is_derived: bool
    derived_from: Sequence[str] = ()
    was_imputed: bool = False
    share_xgb: float = 0.0
    share_gnn: Optional[float] = None
    share_blended: float = 0.0
    #: signed raw values, kept for debugging/audit, not for display
    raw_xgb: float = 0.0
    raw_gnn: Optional[float] = None


@dataclass
class PatientExplanation:
    """Everything the presentation layer needs to explain one verdict."""

    patient_id: str
    decision_code: int
    etiology: str
    explained_class: int
    explained_label: str
    alpha: float
    proba: Dict[str, float] = field(default_factory=dict)
    proba_xgb: Dict[str, float] = field(default_factory=dict)
    proba_gnn: Dict[str, float] = field(default_factory=dict)
    p_b12_conditional: Optional[float] = None
    grey_zone_band: Optional[tuple] = None
    gnn_available: bool = False
    imputed_features: Sequence[str] = ()
    contributions: List[FeatureContribution] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)

    def top(self, limit: int = 3) -> List[FeatureContribution]:
        """Largest contributors first, by absolute blended share."""
        return sorted(
            self.contributions, key=lambda c: abs(c.share_blended), reverse=True
        )[:limit]


def _to_shares(values: Dict[str, float]) -> Dict[str, float]:
    """Convert raw contributions to fractions of total absolute contribution.

    Signed, so a feature that pushed *against* the explained class keeps a
    negative share rather than being folded in as if it supported it.
    """
    total = float(sum(abs(v) for v in values.values()))
    if total <= 0.0:
        return {k: 0.0 for k in values}
    return {k: v / total for k, v in values.items()}


def _explained_class(decision_code: int, proba: np.ndarray) -> int:
    """Which class the explanation should describe.

    For a committed verdict this is the verdict itself. Grey-zone triage (code
    4) has no committed etiology, so we explain whichever deficiency class the
    model ranked highest -- the question there is "what nearly tipped this?",
    not "what was decided?".
    """
    if 0 <= decision_code <= 3:
        return decision_code
    deficiency_probs = [(int(k), float(proba[k])) for k in range(1, 4)]
    return max(deficiency_probs, key=lambda kv: kv[1])[0]


def explain_patient(
    grader: Any,
    row: Dict[str, Any],
    imputed_features: Optional[Set[str]] = None,
    include_gnn: bool = True,
    steps: int = DEFAULT_STEPS,
) -> PatientExplanation:
    """Explain one patient's verdict from the deployed hybrid model.

    ``row`` is a single record of the 9 base values (missing allowed as ``None``),
    exactly what ``model_client.grade_record`` feeds the model.

    ``imputed_features`` is the set of biomarker keys the pipeline had to impute
    (``GraderInputBuilder`` records these under ``imputed``). Pass it so the
    explanation can mark contributors that rest on values the patient never
    actually had measured -- an attribution pointing at a guessed ferritin is
    worse than no attribution, because it reads as evidence.

    The verdict is *not* computed here; it is read from the grader, so this can
    never explain a different decision than the one being shown.
    """
    rows = [row]
    probs = grader.predict_proba(rows)

    decision = int(probs["decision"][0])
    p_blend = np.asarray(probs["proba_blended"][0], dtype=float)
    p_xgb = np.asarray(probs["proba_xgb"][0], dtype=float)
    p_gnn = np.asarray(probs["proba_gnn"][0], dtype=float)

    denom = float(p_blend[1] + p_blend[2])
    p_b12_cond = (float(p_blend[2]) / denom) if denom > 0 else None

    lower = float(getattr(grader, "lower_bound", 0.25))
    upper = float(getattr(grader, "upper_bound", 0.75))
    in_grey_band = p_b12_cond is not None and lower <= p_b12_cond <= upper

    explained_class = _explained_class(decision, p_blend)
    alpha = float(getattr(grader, "alpha", 0.5))

    caveats: List[str] = []
    imputed = {str(k) for k in (imputed_features or set())}

    # ---- branch 1: XGBoost cascade, exact per stage, composed by chain rule
    imputed_frame, scaled_frame = grader.preprocess(rows)
    X_imp = imputed_frame.to_numpy(dtype=float)
    stage_results = explain_xgb(grader.cascade, X_imp)[0]
    xgb_values = compose_class_attribution(stage_results, explained_class)

    # ---- branch 2: hetero-GNN, Integrated Gradients on the patient node
    gnn_values: Optional[Dict[str, float]] = None
    gnn_available = False
    if include_gnn:
        try:
            gnn_result = explain_gnn(
                grader.gnn,
                scaled_frame,
                scaler=getattr(grader, "scaler", None),
                steps=steps,
                class_indices=[explained_class],
            )[0]
            gnn_values = dict(gnn_result.scaled_contributions)
            gnn_available = True
        except (RuntimeError, ImportError) as exc:
            caveats.append(f"GNN branch excluded from this explanation: {exc}")

    if not gnn_available:
        caveats.append(
            "Contributions reflect the XGBoost cascade alone. The deployed model "
            "blends it with the GNN, so these shares are not the full model's."
        )

    # ---- blend on shares, not on raw units (see module docstring)
    share_xgb = _to_shares(xgb_values)
    share_gnn = _to_shares(gnn_values) if gnn_values else None

    blended_shares: Dict[str, float] = {}
    for name in FEATURE_NAMES:
        if share_gnn is not None:
            blended_shares[name] = alpha * share_xgb.get(name, 0.0) + (
                1.0 - alpha
            ) * share_gnn.get(name, 0.0)
        else:
            blended_shares[name] = share_xgb.get(name, 0.0)

    # Renormalise after blending. Each branch's shares sum to 1 on their own, but
    # the blend is signed: a feature the cascade credits and the GNN debits nets
    # toward zero, so the blended list sums to well under 1 without this step. A
    # UI rendering these as "share of contribution" would then print percentages
    # that do not add up, which reads as a bug long before anyone works out why.
    blended_shares = _to_shares(blended_shares)

    contributions: List[FeatureContribution] = []
    for name in FEATURE_NAMES:
        contributions.append(
            FeatureContribution(
                feature=name,
                label=FEATURE_LABELS.get(name, name),
                is_derived=name in DERIVED_PARENTS,
                derived_from=DERIVED_PARENTS.get(name, ()),
                was_imputed=name in imputed,
                share_xgb=float(share_xgb.get(name, 0.0)),
                share_gnn=float(share_gnn[name]) if share_gnn is not None else None,
                share_blended=float(blended_shares.get(name, 0.0)),
                raw_xgb=float(xgb_values.get(name, 0.0)),
                raw_gnn=float(gnn_values[name]) if gnn_values is not None else None,
            )
        )

    caveats.append(
        "The two branches are explained by different methods (TreeSHAP and "
        "Integrated Gradients) with different units, so their contributions are "
        "blended as normalised shares rather than as the model's own alpha blend."
    )
    if any(s.is_calibrated for s in stage_results):
        caveats.append(
            "One cascade stage is isotonic-calibrated; it is explained before "
            "calibration, so the monotone transform that follows is not "
            "attributed per feature."
        )
    imputed_contributors = [c.label for c in contributions if c.was_imputed]
    if imputed and imputed_contributors:
        caveats.append(
            "Some values were estimated rather than measured ("
            + ", ".join(sorted(set(imputed_contributors)))
            + "); contributions resting on them are marked."
        )

    return PatientExplanation(
        patient_id=str(row.get("patient_id", "")),
        decision_code=decision,
        etiology=DECISION_NAMES[decision],
        explained_class=explained_class,
        explained_label=ATTRIBUTION_CLASSES[explained_class],
        alpha=alpha,
        proba={n: float(v) for n, v in zip(ATTRIBUTION_CLASSES, p_blend)},
        proba_xgb={n: float(v) for n, v in zip(ATTRIBUTION_CLASSES, p_xgb)},
        proba_gnn={n: float(v) for n, v in zip(ATTRIBUTION_CLASSES, p_gnn)},
        p_b12_conditional=p_b12_cond,
        grey_zone_band=(lower, upper) if in_grey_band else None,
        gnn_available=gnn_available,
        imputed_features=sorted(imputed),
        contributions=contributions,
        caveats=caveats,
    )


def base_feature_rollup(explanation: PatientExplanation) -> Dict[str, float]:
    """Blended shares for the 9 *measured* values only.

    The 12 model features include three indices computed from the other nine
    (MCV/RBC and friends). Reporting all twelve double-counts: a patient with a
    low MCV shows up once as MCV and again as the Mentzer index built from it.
    This rollup drops the derived features so a headline list names only values
    the patient actually had measured.
    """
    return {
        c.feature: c.share_blended
        for c in explanation.contributions
        if c.feature in BASE_NAMES
    }
