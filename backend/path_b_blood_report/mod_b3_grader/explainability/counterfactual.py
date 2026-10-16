"""Level 2: counterfactual explanations.

Level 1 (:mod:`attributions`) decomposes the verdict the model already reached.
This module answers the inverse question: **what would have to change for the
verdict to move, and by how much?** For each measured biomarker it searches for
the value at which the decision flips, and reports the nearest boundary --
or reports that nothing within plausibility moves it, which is a real answer.

What this deliberately does not claim
-------------------------------------
Each result is **univariate**: one biomarker is varied while every other value
is held fixed. Real biomarkers do not move independently -- ferritin and
haemoglobin rise and fall together in iron deficiency -- so "if ferritin alone
reached 26 ng/mL" describes a point the model can be queried at, not a patient
that exists. The output says so on every result. Treating these numbers as
clinical advice would be a category error, and the caveat list exists to make
that hard to do by accident.

Imputed features are refused rather than searched. Asking what ferritin value
would flip a verdict, on a report where ferritin was never measured, invites
counselling on a number nobody has; the honest answer is that the report cannot
support the question.

Every reported flip is verified by re-running the model on the proposed row
before it is returned. The search converges on a crossing point, and a search
can converge on an artefact; the verification is what makes the claim safe.

Cost
----
The GNN shares its 12 biomarker nodes across a batch and aggregates
``patient -> biomarker`` (see ``grader_model/gnn.py``), so a patient's logits
depend on who else is in the batch. Deployed inference is always batch-1, so
every probe here is batch-1 too -- no batching the grid, even though it would be
much faster. Expect low seconds per patient; callers should cache.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from ..grader_model.features import BASE_NAMES
from .attributions import DECISION_NAMES, FEATURE_LABELS

#: Grid resolution per feature before any bisection.
DEFAULT_SAMPLES = 21
#: Bisection stops when the bracket is this fraction of the searched span.
DEFAULT_TOL_FRAC = 1e-3
#: A flip this close to a range edge means the true boundary may lie outside.
BOUNDARY_EPS_FRAC = 1e-6

#: Clinically plausible adult ranges, in the units the grader consumes
#: (g/dL, fL, %, ng/mL, pg/mL, x10^12/L).
#:
#: These are **hand-set from clinical reference limits, not learned from the
#: model's training data** -- this package never sees that data. They are wide
#: enough that the true boundary is normally interior, and narrow enough to
#: exclude absurdities. Model training support is unknown here and is almost
#: certainly narrower, so a flip found near an edge is a statement about
#: extrapolation. ``at_range_boundary`` marks those.
PLAUSIBLE_RANGES: Dict[str, tuple] = {
    "hemoglobin": (3.0, 20.0),
    "mcv": (50.0, 130.0),
    "rdw": (10.0, 30.0),
    "ferritin": (1.0, 500.0),
    "b12": (50.0, 1500.0),
    "folate": (0.5, 30.0),
    "rbc": (1.5, 7.0),
    "mch": (15.0, 45.0),
    "mchc": (25.0, 40.0),
}

UNIVARIATE_CAVEAT = (
    "Each figure changes one biomarker while holding the others fixed. "
    "Biomarkers move together in real patients, so these describe the model's "
    "boundary, not a course a patient could follow."
)
RANGE_CAVEAT = (
    "The search is bounded by hand-set clinical ranges, not by the model's "
    "training support, which this package cannot see. A boundary found near an "
    "edge of that range may reflect extrapolation rather than biology."
)


#: Reporting units for the nine base values, as printed on Indian lab reports.
#: Carried alongside every boundary: a bare "26.4" for ferritin is ambiguous
#: enough to be read as a different assay entirely.
FEATURE_UNITS: Dict[str, str] = {
    "hemoglobin": "g/dL",
    "mcv": "fL",
    "rdw": "%",
    "ferritin": "ng/mL",
    "b12": "pg/mL",
    "folate": "ng/mL",
    "rbc": "10^12/L",
    "mch": "pg",
    "mchc": "g/dL",
}


@dataclass
class Flip:
    """The nearest value of one biomarker at which the verdict changes."""

    feature: str
    label: str
    unit: str
    current_value: float
    target_value: float
    direction: str  # "increase" | "decrease"
    to_decision: int
    to_etiology: str
    verified: bool
    at_range_boundary: bool = False

    @property
    def delta(self) -> float:
        return self.target_value - self.current_value


@dataclass
class CounterfactualReport:
    """Nearest-flip search across every measured biomarker."""

    patient_id: str
    decision_code: int
    etiology: str
    target_decision: Optional[int] = None
    flips: List[Flip] = field(default_factory=list)
    unreachable: List[str] = field(default_factory=list)
    refused: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)

    def nearest(self) -> Optional[Flip]:
        """The smallest change that moves the verdict, or None."""
        if not self.flips:
            return None
        return min(self.flips, key=lambda f: abs(f.delta))


def _decision(grader: Any, row: Dict[str, Any]) -> int:
    """The deployed decision for one row. Always batch-1 (see module docstring)."""
    return int(grader.predict_proba([row])["decision"][0])


def _is_missing(value: Any) -> bool:
    """A value the model would have to impute.

    ``None`` is what ``GraderInputBuilder`` produces for an unmeasured marker,
    but callers and the test corpus also pass ``NaN``. Both mean "not measured",
    and a counterfactual must refuse both rather than search over a number the
    patient never had.
    """
    if value is None:
        return True
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _bisect_flip(
    grader: Any,
    row: Dict[str, Any],
    feature: str,
    base_decision: int,
    a: float,
    b: float,
    tol: float,
    target_decision: Optional[int],
) -> Optional[float]:
    """Narrow a bracket [a, b] known to contain a decision change.

    ``a`` must sit on the base decision and ``b`` on a different one; the
    returned value is the first point, within ``tol``, that achieves the flip.
    Endpoint order does not matter -- the bracket is only ever shrunk inward.
    """
    for _ in range(64):  # generous cap; tol normally stops this far sooner
        if abs(b - a) <= tol:
            break
        mid = 0.5 * (a + b)
        probe = dict(row)
        probe[feature] = float(mid)
        if target_decision is None:
            hit = _decision(grader, probe) != base_decision
        else:
            # When a specific destination is requested, only that class counts
            # as a hit; passing *through* another class on the way is not one.
            hit = _decision(grader, probe) == target_decision
        if hit:
            b = mid
        else:
            a = mid
    return float(b)


def _search_feature(
    grader: Any,
    row: Dict[str, Any],
    feature: str,
    base_value: float,
    base_decision: int,
    samples: int,
    tol_frac: float,
    target_decision: Optional[int],
) -> Optional[float]:
    """Smallest change in ``feature`` that reaches a different decision.

    Grid first, then bisect. A pure bisection would assume the decision changes
    monotonically along the line; it need not -- the triage rules are a chain of
    thresholds on blended probabilities -- so the grid establishes where the
    crossings actually are before anything is narrowed.
    """
    lo, hi = PLAUSIBLE_RANGES[feature]
    if not (lo <= base_value <= hi):
        # The patient sits outside the range we are willing to search. Clamp so
        # the grid still covers the plausible side rather than skipping them.
        lo = min(lo, base_value)
        hi = max(hi, base_value)

    grid = np.linspace(lo, hi, samples)
    decisions = np.empty(len(grid), dtype=int)
    for i, value in enumerate(grid):
        probe = dict(row)
        probe[feature] = float(value)
        decisions[i] = _decision(grader, probe)

    def _is_hit(d: int) -> bool:
        return d == target_decision if target_decision is not None else d != base_decision

    hits = [i for i in range(len(grid)) if _is_hit(int(decisions[i]))]
    if not hits:
        return None

    # Nearest hit to the patient's own value, so the reported boundary is the
    # closest one rather than whichever the grid happened to visit first.
    j = min(hits, key=lambda i: abs(grid[i] - base_value))
    direction = 1.0 if grid[j] >= base_value else -1.0

    # Walk back toward the patient's value to find the near side of the bracket.
    k = j
    while True:
        nxt = k - int(direction)
        if nxt < 0 or nxt >= len(grid):
            # The hit sits at the edge of the grid: there is no base-decision
            # neighbour to bracket against, so report the edge point itself.
            # It was a verified hit on the grid pass, just a coarse one --
            # ``at_range_boundary`` flags it downstream rather than pretending
            # no boundary exists.
            return float(grid[j])
        if not _is_hit(int(decisions[nxt])):
            k = nxt
            break
        k = nxt

    return _bisect_flip(
        grader, row, feature, base_decision,
        float(grid[k]), float(grid[j]), tol_frac * abs(hi - lo), target_decision,
    )


def counterfactual_flips(
    grader: Any,
    row: Dict[str, Any],
    imputed_features: Optional[Set[str]] = None,
    target_decision: Optional[int] = None,
    samples: int = DEFAULT_SAMPLES,
    tol_frac: float = DEFAULT_TOL_FRAC,
) -> CounterfactualReport:
    """Find, per measured biomarker, the nearest value that moves the verdict.

    ``row`` is a single record of the 9 base values, exactly what
    ``model_client.grade_record`` feeds the model. ``imputed_features`` names
    the biomarkers the pipeline had to guess; those are refused, not searched.

    ``target_decision`` restricts the search to one destination class. Left as
    ``None``, the search reports the nearest boundary of *any* kind, which is
    usually the more useful question -- "how far from any change in verdict is
    this patient?" -- and always names which verdict is on the other side.
    """
    base_decision = _decision(grader, row)
    imputed = {str(k) for k in (imputed_features or set())}
    flips: List[Flip] = []
    unreachable: List[str] = []
    refused: List[str] = []

    for feature in BASE_NAMES:
        raw = row.get(feature)
        if feature in imputed or _is_missing(raw):
            refused.append(feature)
            continue
        if feature not in PLAUSIBLE_RANGES:
            refused.append(feature)
            continue

        base_value = float(raw)
        target = _search_feature(
            grader, row, feature, base_value, base_decision,
            samples, tol_frac, target_decision,
        )
        if target is None:
            unreachable.append(feature)
            continue

        # Verify before reporting. The search converges on a crossing; this is
        # what establishes that the crossing is real for the deployed model.
        probe = dict(row)
        probe[feature] = float(target)
        landed = _decision(grader, probe)
        if landed == base_decision:
            unreachable.append(feature)
            continue
        if target_decision is not None and landed != target_decision:
            # The bisection stopped somewhere other than the requested class.
            # Reporting it as a route to ``target_decision`` would be a claim
            # the model does not support.
            unreachable.append(feature)
            continue

        lo, hi = PLAUSIBLE_RANGES[feature]
        span = abs(hi - lo) or 1.0
        flips.append(
            Flip(
                feature=feature,
                label=FEATURE_LABELS.get(feature, feature),
                unit=FEATURE_UNITS.get(feature, ""),
                current_value=base_value,
                target_value=float(target),
                direction="increase" if target > base_value else "decrease",
                to_decision=landed,
                to_etiology=DECISION_NAMES[landed],
                verified=True,
                at_range_boundary=(
                    abs(float(target) - lo) <= BOUNDARY_EPS_FRAC * span
                    or abs(float(target) - hi) <= BOUNDARY_EPS_FRAC * span
                ),
            )
        )

    flips.sort(key=lambda f: abs(f.delta))

    caveats = [UNIVARIATE_CAVEAT, RANGE_CAVEAT]
    if refused:
        caveats.append(
            "Not searched because the value was estimated rather than measured: "
            + ", ".join(FEATURE_LABELS.get(f, f) for f in refused)
            + ". The report cannot support a question about a test it did not run."
        )
    if target_decision is not None:
        caveats.append(
            f"The search was restricted to {DECISION_NAMES[target_decision]}. "
            "A boundary of a different class may be closer."
        )
    if any(f.at_range_boundary for f in flips):
        caveats.append(
            "At least one boundary sits at the edge of the searched range, so "
            "the true crossing may lie beyond it."
        )

    return CounterfactualReport(
        patient_id=str(row.get("patient_id", "")),
        decision_code=base_decision,
        etiology=DECISION_NAMES[base_decision],
        target_decision=target_decision,
        flips=flips,
        unreachable=unreachable,
        refused=refused,
        caveats=caveats,
    )
