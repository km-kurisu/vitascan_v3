"""Exact per-stage SHAP attributions for the XGBoost cascade (Level 1, XGB half).

Why per-stage first
-------------------
Each cascade stage is a gradient-boosted tree ensemble, where TreeSHAP is
*exact*: the per-feature values sum to the stage's own output. The three stages
are then combined multiplicatively (see ``grader_model/cascade.py``)::

    p0 = 1 - a
    p1 = a * (1 - f) * (1 - b)
    p2 = a * (1 - f) * b
    p3 = a * f

with ``a = s(m1)``, ``f = s(m2a)``, ``b = s(m2b)`` and ``s`` the logistic
function. Taking logs turns each product into a sum of ``log s(...)`` terms --
but ``log s`` is *not* linear in the margin, so simply adding the three stages'
SHAP values does **not** give the exact attribution of ``log p_k``.

So this module is exact where it can be and explicit where it cannot:

* per-stage SHAP values are exact (TreeSHAP's efficiency axiom holds per stage);
* combining them into a class attribution uses the chain rule at the patient's
  operating point, which is a first-order approximation;
* :func:`composition_residual` measures that approximation against the cascade
  itself, so the error is quantified rather than assumed. Callers that need an
  exact end-to-end number should use full-enumeration KernelSHAP instead.

Calibration caveat (stage2b)
----------------------------
``stage2b_calibrated`` is an isotonic-calibrated wrapper. Isotonic regression is
piecewise-constant, so its derivative is zero almost everywhere and TreeSHAP
cannot see through it. We therefore unwrap to the tree underneath and explain
its *pre-calibration margin*. That is the honest choice -- the attributions
describe the tree, and the monotone calibration that follows is not attributed
per feature. ``StageShapResult.is_calibrated`` marks this for callers so the UI
can say so rather than implying the number already accounts for calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np

from ..grader_model.cascade import ETIOLOGY_NAMES
from ..grader_model.features import FEATURE_NAMES

# The four etiology classes the cascade votes over, in the model's own order.
ATTRIBUTION_CLASSES: List[str] = list(ETIOLOGY_NAMES)

# Stage attribute names on XGBCascadePredictor, mapped to what they decide.
STAGE_LABELS: Dict[str, str] = {
    "stage1_xgb": "any anemia vs none",
    "stage2a_xgb": "folate vs IDA/B12",
    "stage2b_calibrated": "B12 vs IDA (pre-calibration)",
}

# Which stages are wrapped in a monotone calibrator (not attributed per-feature).
_CALIBRATED_STAGES = frozenset({"stage2b_calibrated"})


@dataclass
class StageShapResult:
    """Exact SHAP values for one cascade stage, in that stage's margin space."""

    stage: str
    label: str
    is_calibrated: bool
    margin: float
    base_value: float
    #: feature name -> contribution to this stage's margin (log-odds units)
    contributions: Dict[str, float] = field(default_factory=dict)

    def summed(self) -> float:
        """Sum of contributions; equals ``margin - base_value`` by efficiency."""
        return float(sum(self.contributions.values()))


def _require_shap():
    """Import SHAP lazily so a missing dependency never breaks the pipeline."""
    try:
        import shap  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Explainability requires the 'shap' package. Install it with "
            "`pip install shap` (and add it to backend/requirements.txt). "
            "The model itself does not depend on SHAP and keeps working without it."
        ) from exc
    return shap


def unwrap_calibrated(estimator: Any) -> Any:
    """Return the tree ensemble underneath a CalibratedClassifierCV, if wrapped.

    Handles both the modern ``calibrated_classifiers_[i].estimator`` layout and
    the legacy ``base_estimator`` attribute. Non-calibrated estimators are
    returned unchanged, so this is safe to call unconditionally.
    """
    calibrated = getattr(estimator, "calibrated_classifiers_", None)
    if not calibrated:
        return estimator
    first = calibrated[0]
    for attr in ("estimator", "base_estimator"):
        inner = getattr(first, attr, None)
        if inner is not None:
            return inner
    return estimator


def _positive_margin(estimator: Any, X: np.ndarray) -> np.ndarray:
    """Positive-class margin (log-odds) for each row.

    Preferred over reconstructing the margin from SHAP's base value plus
    contributions, so that any disagreement between the two shows up as a
    residual instead of being silently hidden.
    """
    try:
        return np.asarray(estimator.predict(X, output_margin=True), dtype=float).ravel()
    except TypeError:
        import xgboost as xgb  # noqa: PLC0415

        booster = estimator.get_booster()
        return np.asarray(
            booster.predict(xgb.DMatrix(X), output_margin=True), dtype=float
        ).ravel()


def _tree_explainer(shap_module: Any, estimator: Any) -> Any:
    """Build a TreeExplainer pinned to raw-margin output.

    ``model_output`` is a *constructor* argument on ``TreeExplainer``; passing it
    to ``shap_values`` raises TypeError. Some builds also reject the kwarg on the
    constructor, in which case we fall back to the default -- for XGBoost that is
    already the raw margin, which ``test_stage_shap_is_exact`` verifies against
    the margin computed independently.
    """
    try:
        return shap_module.TreeExplainer(estimator, model_output="raw")
    except TypeError:
        return shap_module.TreeExplainer(estimator)


def _shap_matrix(estimator: Any, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-row SHAP values for the positive class, plus the explainer's base value.

    SHAP's return shape for binary classifiers has changed across versions
    (list-of-two, ``(n, f)``, or ``(n, f, classes)``), so normalise all of them
    here rather than letting a version bump produce silently wrong numbers.
    """
    shap_module = _require_shap()
    explainer = _tree_explainer(shap_module, estimator)
    raw = explainer.shap_values(X)

    if isinstance(raw, list):
        raw = raw[1] if len(raw) == 2 else raw[0]
    values = np.asarray(raw, dtype=float)
    if values.ndim == 3:
        # (n_rows, n_features, n_classes) -> positive class
        values = values[:, :, 1]
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Unexpected SHAP output shape {values.shape} for a "
            f"{len(FEATURE_NAMES)}-feature model; cannot attribute safely."
        )

    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base_arr = np.asarray(base, dtype=float).ravel()
        base_value = float(base_arr[1] if base_arr.size > 1 else base_arr[0])
    else:
        base_value = float(base)
    return values, base_value


def explain_xgb(cascade: Any, X_imputed: np.ndarray) -> List[List[StageShapResult]]:
    """Explain the cascade for every row of ``X_imputed``.

    ``X_imputed`` is the KNN-imputed, **unscaled** 12-feature matrix -- the
    cascade is trained on imputed features, while the GNN is trained on scaled
    ones (see ``AnemiaGrader.predict_proba``). Passing the scaled matrix here
    would produce plausible-looking but meaningless attributions.

    Returns one list of :class:`StageShapResult` per input row, ordered as
    :data:`STAGE_LABELS`.
    """
    X = np.asarray(X_imputed, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} features, got {X.shape[1]}. "
            "Order must match FEATURE_NAMES."
        )

    per_row: List[List[StageShapResult]] = [[] for _ in range(len(X))]

    for stage in STAGE_LABELS:
        wrapper = getattr(cascade, stage)
        tree = unwrap_calibrated(wrapper)
        values, base_value = _shap_matrix(tree, X)
        margins = _positive_margin(tree, X)
        is_calibrated = stage in _CALIBRATED_STAGES

        for i in range(len(X)):
            per_row[i].append(
                StageShapResult(
                    stage=stage,
                    label=STAGE_LABELS[stage],
                    is_calibrated=is_calibrated,
                    margin=float(margins[i]),
                    base_value=base_value,
                    contributions={
                        name: float(values[i, j]) for j, name in enumerate(FEATURE_NAMES)
                    },
                )
            )

    return per_row


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def chain_coefficients(m1: float, m2a: float, m2b: float) -> Dict[int, Dict[str, float]]:
    """``d(log p_k) / d(stage margin)`` evaluated at this patient's margins.

    Derived by differentiating the log-product form in the module docstring::

        log p0 = log s(-m1)
        log p1 = log s(m1) + log s(-m2a) + log s(-m2b)
        log p2 = log s(m1) + log s(-m2a) + log s(m2b)
        log p3 = log s(m1) + log s(m2a)

    Each derivative is a sigmoid of the relevant margin, so these are exact
    locally -- they are the first-order expansion of the true attribution.
    """
    s_m1 = float(_sigmoid(m1))
    s_m2a = float(_sigmoid(m2a))
    s_m2b = float(_sigmoid(m2b))
    return {
        0: {"stage1_xgb": -s_m1, "stage2a_xgb": 0.0, "stage2b_calibrated": 0.0},
        1: {
            "stage1_xgb": 1.0 - s_m1,
            "stage2a_xgb": -s_m2a,
            "stage2b_calibrated": -s_m2b,
        },
        2: {
            "stage1_xgb": 1.0 - s_m1,
            "stage2a_xgb": -s_m2a,
            "stage2b_calibrated": 1.0 - s_m2b,
        },
        3: {
            "stage1_xgb": 1.0 - s_m1,
            "stage2a_xgb": 1.0 - s_m2a,
            "stage2b_calibrated": 0.0,
        },
    }


def compose_class_attribution(
    stages: Sequence[StageShapResult], class_index: int
) -> Dict[str, float]:
    """Fold three stage results into per-feature attributions for one class.

    The returned values are in **log-probability units**: they approximate how
    each biomarker moved ``log p_class`` away from its baseline. They are the
    first-order composition of exact per-stage SHAP values, not exact Shapley
    values of the composed function -- see :func:`composition_residual`.
    """
    if class_index not in (0, 1, 2, 3):
        raise ValueError(f"class_index must be 0..3, got {class_index}")

    by_stage = {s.stage: s for s in stages}
    missing = set(STAGE_LABELS) - set(by_stage)
    if missing:
        raise ValueError(f"Missing stage results: {sorted(missing)}")

    coeffs = chain_coefficients(
        by_stage["stage1_xgb"].margin,
        by_stage["stage2a_xgb"].margin,
        by_stage["stage2b_calibrated"].margin,
    )[class_index]

    totals: Dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}
    for stage_name, coef in coeffs.items():
        if coef == 0.0:
            continue
        for feature, value in by_stage[stage_name].contributions.items():
            totals[feature] += coef * value
    return totals


def composition_residual(
    cascade: Any,
    X_imputed: np.ndarray,
    class_index: int,
    feature_index: int,
    step_frac: float = 1e-3,
) -> float:
    """Measure the composition approximation against the cascade itself.

    Computes the attribution for one feature analytically, then compares it to
    the actual derivative of ``log p_class`` with respect to that imputed
    feature, estimated by a central difference. Returns the absolute gap in
    log-probability units.

    The step is *relative* to the feature's own magnitude, because the imputed
    matrix holds raw clinical values spanning two orders of magnitude --
    hemoglobin ~14 alongside ferritin ~130. A single absolute step would be
    noise for one and a large excursion for the other, and the resulting
    residual would say more about the step size than about the approximation.

    This exists so the approximation is *tested*, not asserted: if the residual
    is not small on the sample corpus, the composition step is not trustworthy
    and the caller should switch to full-enumeration KernelSHAP.
    """
    X = np.asarray(X_imputed, dtype=float).reshape(1, -1)
    stages = explain_xgb(cascade, X)[0]
    analytic = compose_class_attribution(stages, class_index)[FEATURE_NAMES[feature_index]]

    h = step_frac * max(abs(float(X[0, feature_index])), 1e-6)

    def log_prob(matrix: np.ndarray) -> float:
        p = float(cascade.predict_proba(matrix)[0, class_index])
        return float(np.log(max(p, 1e-12)))

    up = X.copy()
    up[0, feature_index] += h
    down = X.copy()
    down[0, feature_index] -= h
    observed = (log_prob(up) - log_prob(down)) / (2.0 * h)
    return abs(analytic - observed)
