"""Tests for the explainability sidecar.

The load-bearing test here is :func:`test_predictions_match_manifest`. The entire
case for building XAI as a sidecar rather than inside the model is that the
verdicts stay byte-identical -- if that test fails, the approach was wrong and
nothing else in this file matters.

Everything except the manifest test needs ``shap``; they skip cleanly without it
rather than erroring, because the model itself does not depend on SHAP.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.path_b_blood_report.mod_b3_grader.grader_model.features import (
    BASE_NAMES,
    FEATURE_NAMES,
)

MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "shared"
    / "sample_reports"
    / "test_pdfs"
    / "MANIFEST.json"
)

#: IND-07 is CBC-only; the pipeline imputes ferritin/B12/folate for it.
IMPUTED_IN_IND07 = {"ferritin", "b12", "folate"}


def _manifest_entries():
    with MANIFEST.open(encoding="utf-8") as handle:
        return json.load(handle)


def _row_for(entry) -> dict:
    """Build a model row from a manifest entry (missing values as NaN)."""
    biochem = entry["biochemistry"]
    return {
        "patient_id": entry["patient_id"],
        **{
            name: (np.nan if biochem.get(name) is None else biochem.get(name))
            for name in BASE_NAMES
        },
    }


def _entry(patient_id: str) -> dict:
    for candidate in _manifest_entries():
        if candidate["patient_id"] == patient_id:
            return candidate
    raise AssertionError(f"{patient_id} missing from MANIFEST.json")


@pytest.fixture(scope="module")
def grader():
    """The deployed model. Heavy to construct (torch + torch_geometric)."""
    from backend.path_b_blood_report.mod_b3_grader.grader_model.predict import (
        AnemiaGrader,
    )

    return AnemiaGrader()


# --------------------------------------------------------------------- guard


def test_predictions_match_manifest(grader):
    """Adding explanations must not have moved a single verdict.

    This is the regression guard for the sidecar approach. If it fails, either
    the explainability package reached into the model, or the model changed
    underneath the corpus -- both worth stopping for.
    """
    mismatches = []
    for entry in _manifest_entries():
        row = _row_for(entry)
        got = int(grader.predict_proba([row])["decision"][0])
        expected = int(entry["expected_decision_code"])
        if got != expected:
            mismatches.append(
                f"{entry['patient_id']}: expected {expected}, got {got}"
            )
    assert not mismatches, "Verdict drift:\n" + "\n".join(mismatches)


def test_grey_zone_corpus_is_reachable(grader):
    """IND-04/IND-06 exist to probe triage; if they stop triaging, the corpus
    no longer exercises the branch the explanations are most useful for."""
    for patient_id in ("IND-04-MEGALOBLASTIC-B12-MALE", "IND-06-MIXED-GREYZONE-MALE"):
        row = _row_for(_entry(patient_id))
        assert int(grader.predict_proba([row])["decision"][0]) == 4


# ------------------------------------------------------------- stage SHAP


def _imputed_matrix(grader, entry) -> np.ndarray:
    imputed, _scaled = grader.preprocess([_row_for(entry)])
    return imputed.to_numpy(dtype=float)


def test_stage_shap_is_exact(grader):
    """TreeSHAP's efficiency axiom: contributions sum to the margin shift."""
    pytest.importorskip("shap")
    from backend.path_b_blood_report.mod_b3_grader.explainability import explain_xgb

    X = _imputed_matrix(grader, _entry("IND-03-IDA-FEMALE"))
    stages = explain_xgb(grader.cascade, X)[0]

    assert [s.stage for s in stages] == [
        "stage1_xgb",
        "stage2a_xgb",
        "stage2b_calibrated",
    ]
    for stage in stages:
        assert stage.summed() == pytest.approx(
            stage.margin - stage.base_value, abs=1e-4
        )


def test_composition_residual_is_small(grader):
    """The chain-rule composition must track the cascade it approximates.

    Threshold is deliberately loose -- this is a first-order method, and the
    point is to catch a *broken* composition (residual of order 1), not to
    assert a precision the maths does not promise.
    """
    pytest.importorskip("shap")
    from backend.path_b_blood_report.mod_b3_grader.explainability import (
        composition_residual,
    )

    X = _imputed_matrix(grader, _entry("IND-03-IDA-FEMALE"))
    residuals = [
        composition_residual(grader.cascade, X, class_index=1, feature_index=j)
        for j, name in enumerate(FEATURE_NAMES)
        if not np.isnan(X[0, j])
    ]
    worst = max(residuals)
    assert worst < 0.25, f"composition residual {worst:.3f} is too large"


# ------------------------------------------------------------------ GNN IG


def test_ig_completeness_holds(grader):
    """Integrated Gradients must sum to the change it claims to explain.

    The tolerance is loose on purpose. IG satisfies completeness only in the
    limit of infinitely many interpolation steps; at 64 steps a small residual is
    expected (~2e-3 against logits of order 1-10). This test exists to catch a
    *broken* implementation -- wrong baseline, wrong target node, gradients not
    accumulating -- which shows up as a gap of order 1, not as 1e-3.
    """
    pytest.importorskip("shap")
    from backend.path_b_blood_report.mod_b3_grader.explainability import (
        completeness_gap,
    )

    _imputed, scaled = grader.preprocess([_row_for(_entry("IND-03-IDA-FEMALE"))])
    assert abs(completeness_gap(grader.gnn, scaled, class_index=1, steps=64)) < 0.01


def test_gnn_attribution_has_twelve_features(grader):
    pytest.importorskip("shap")
    from backend.path_b_blood_report.mod_b3_grader.explainability import explain_gnn

    _imputed, scaled = grader.preprocess([_row_for(_entry("IND-03-IDA-FEMALE"))])
    result = explain_gnn(grader.gnn, scaled, scaler=grader.scaler, class_indices=[1])[0]
    assert set(result.scaled_contributions) == set(FEATURE_NAMES)
    assert set(result.clinical_contributions) == set(FEATURE_NAMES)


# ------------------------------------------------------------ orchestrator


def test_explanation_agrees_with_verdict(grader):
    """The explanation must never describe a different decision than the one shown."""
    pytest.importorskip("shap")
    from backend.path_b_blood_report.mod_b3_grader.explainability.attributions import (
        explain_patient,
    )

    entry = _entry("IND-03-IDA-FEMALE")
    explanation = explain_patient(grader, _row_for(entry))
    assert explanation.decision_code == entry["expected_decision_code"] == 1
    assert explanation.etiology == "IDA"
    assert explanation.explained_label == "IDA"


def test_shares_are_normalised(grader):
    pytest.importorskip("shap")
    from backend.path_b_blood_report.mod_b3_grader.explainability.attributions import (
        explain_patient,
    )

    explanation = explain_patient(grader, _row_for(_entry("IND-03-IDA-FEMALE")))
    total = sum(abs(c.share_blended) for c in explanation.contributions)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_grey_zone_explains_a_candidate_not_a_commitment(grader):
    """IND-04 is a textbook B12 picture the model declines to commit on.

    The explanation should say which deficiency nearly won -- that is the whole
    value of explaining a triage verdict.
    """
    pytest.importorskip("shap")
    from backend.path_b_blood_report.mod_b3_grader.explainability.attributions import (
        explain_patient,
    )

    explanation = explain_patient(grader, _row_for(_entry("IND-04-MEGALOBLASTIC-B12-MALE")))
    assert explanation.decision_code == 4
    assert explanation.etiology == "Grey Zone Triage"
    assert explanation.explained_label in {"IDA", "B12 Deficiency", "Folate Deficiency"}
    assert explanation.grey_zone_band == (grader.lower_bound, grader.upper_bound)
    assert explanation.p_b12_conditional is not None
    assert explanation.top(3), "grey-zone explanation produced no contributors"


def test_imputed_contributors_are_marked(grader):
    """IND-07 never had ferritin/B12/folate measured.

    An explanation that leans on a guessed value must say so; an attribution
    pointing at an imputed ferritin reads as evidence the patient never gave.
    """
    pytest.importorskip("shap")
    from backend.path_b_blood_report.mod_b3_grader.explainability.attributions import (
        explain_patient,
    )

    explanation = explain_patient(
        grader,
        _row_for(_entry("IND-07-CBC-ONLY-FEMALE")),
        imputed_features=IMPUTED_IN_IND07,
    )
    assert set(explanation.imputed_features) == IMPUTED_IN_IND07

    marked = {c.feature for c in explanation.contributions if c.was_imputed}
    assert marked == IMPUTED_IN_IND07
    assert any("estimated rather than measured" in c for c in explanation.caveats)


def test_derived_features_are_flagged_and_rolled_up(grader):
    pytest.importorskip("shap")
    from backend.path_b_blood_report.mod_b3_grader.explainability.attributions import (
        base_feature_rollup,
        explain_patient,
    )

    explanation = explain_patient(grader, _row_for(_entry("IND-03-IDA-FEMALE")))
    derived = [c for c in explanation.contributions if c.is_derived]
    assert {c.feature for c in derived} == {
        "srivastava_index",
        "mentzer_index",
        "cell_hb_density",
    }
    for contribution in derived:
        assert contribution.derived_from, "derived feature has no parents recorded"

    rollup = base_feature_rollup(explanation)
    assert set(rollup) == set(BASE_NAMES), "rollup should cover only measured values"
