"""Baseline model comparison, feeding ``ModB3Output.baseline_comparison``.

This module previously returned a hardcoded literal -- ``{"iron": {"gat": 0.75,
"logistic_regression": 0.68, "random_forest": 0.72, "xgboost": 0.74}}`` -- while
ignoring its argument entirely. It was never an evaluation: no train/test split,
no fitted baseline, no scoring. In parallel, the frontend rendered its own
hardcoded literals (65.0/69.0/70.0) under a "Baseline Models Benchmark" heading,
and the two disagreed with each other. Between them the product implied the
deployed grader had been measured against three baselines. It had not.

Returning ``{}`` is the honest answer: no baseline has been evaluated, so there
is nothing to report. Populate this only from a real evaluation on a held-out
split, and record which split and which metric.
"""

BASELINE_EVALUATION_NOTE = (
    "No baseline evaluation has been run, so no comparison scores exist. This "
    "mapping is empty by design: an unmeasured figure must never reach the UI "
    "looking like a benchmark result."
)


class BaselineModelsEvaluator:
    def predict(self, normalized_biomarkers):
        """Per-deficiency baseline scores, or ``{}`` when none have been measured.

        Deliberately ignores ``normalized_biomarkers``. Baseline comparison
        scores are aggregate metrics for a trained model on an evaluation set,
        not per-patient predictions -- they do not vary with the input row. The
        argument is kept so the call site stays unchanged; a future real
        implementation would load persisted results rather than compute
        anything from the row either.
        """
        return {}
