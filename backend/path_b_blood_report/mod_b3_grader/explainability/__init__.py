"""Explainability sidecar for the anemia-workup hybrid model.

This package *reads* the deployed model — it never modifies it. Predictions must
stay byte-identical to what ``grader_model`` produced before this package
existed, and the 7-report corpus in ``shared/sample_reports/test_pdfs`` pins
that behaviour.

Layout
------
``stage_attribution``  exact per-stage SHAP for the XGBoost cascade
``gnn_attribution``    Integrated Gradients through the hetero-GNN patient node
``counterfactual``     Level 2: nearest value that moves the verdict

Level 1 answers "which biomarkers drove this verdict?". Level 2 answers the
inverse -- "what would have to change for it to move, and by how much?" -- and
is univariate, so it describes the model's boundary rather than a course a
patient could follow. Its results carry that limit in ``caveats``.

Nothing here imports ``grader_model``'s private state at construction time; the
caller passes in a live ``AnemiaGrader`` (see ``model_client.get_grader``), so
``alpha``, ``lower_bound`` and ``upper_bound`` are always read from the model
itself rather than copied. That is deliberate: a stale copy of ``alpha`` is the
single easiest way to make explanations silently disagree with the verdict.
"""
from __future__ import annotations

from .attributions import (
    FeatureContribution,
    PatientExplanation,
    base_feature_rollup,
    explain_patient,
)
from .counterfactual import (
    CounterfactualReport,
    Flip,
    counterfactual_flips,
)
from .gnn_attribution import (
    GnnAttribution,
    completeness_gap,
    explain_gnn,
)
from .stage_attribution import (
    ATTRIBUTION_CLASSES,
    StageShapResult,
    compose_class_attribution,
    composition_residual,
    explain_xgb,
)

__all__ = [
    "ATTRIBUTION_CLASSES",
    "CounterfactualReport",
    "FeatureContribution",
    "Flip",
    "GnnAttribution",
    "PatientExplanation",
    "StageShapResult",
    "base_feature_rollup",
    "completeness_gap",
    "compose_class_attribution",
    "composition_residual",
    "counterfactual_flips",
    "explain_gnn",
    "explain_patient",
    "explain_xgb",
]
