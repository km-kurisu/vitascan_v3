"""Integration tests: explanation wired into the grading path.

The load-bearing test here is :func:`test_explanation_does_not_move_the_verdict`.
Attribution is a reporting feature bolted onto a clinical verdict, and the one
thing that must never happen is for turning it on to change what the model
decided. Everything else in this file is about the explanation being *honest*
once it exists -- labelled with its source, marked when it rests on imputed
values, and absent rather than faked when it cannot be produced.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.path_b_blood_report.mod_b3_grader import model_client
from backend.path_b_blood_report.mod_b3_grader.grader_model.features import BASE_NAMES
from backend.path_b_blood_report.mod_b3_grader.grader_model.predict import (
    DECISION_NAMES,
)
from backend.shared.schemas import (
    BloodParameter,
    FeatureAttribution,
    ModelExplanation,
)

MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "shared"
    / "sample_reports"
    / "test_pdfs"
    / "MANIFEST.json"
)


def _entries():
    with MANIFEST.open(encoding="utf-8") as handle:
        return json.load(handle)


def _entry(patient_id: str) -> dict:
    for candidate in _entries():
        if candidate["patient_id"] == patient_id:
            return candidate
    raise AssertionError(f"{patient_id} missing from MANIFEST.json")


def _record(patient_id: str) -> dict:
    """A GraderInputBuilder-shaped record, as the pipeline would hand it over."""
    biochem = _entry(patient_id)["biochemistry"]
    model_input = {name: biochem.get(name) for name in BASE_NAMES}
    return {
        "patient_id": patient_id,
        "na_count": sum(1 for value in model_input.values() if value is None),
        "model_input": model_input,
        "imputed": None,
    }


@pytest.fixture(scope="module")
def grader():
    """One model load shared by every test in this file (it is expensive)."""
    from backend.path_b_blood_report.mod_b3_grader.grader_model.predict import (
        AnemiaGrader,
    )

    instance = AnemiaGrader()
    model_client._GRADER = instance  # let grade_record reuse it
    yield instance
    model_client._GRADER = None


@pytest.fixture(scope="module")
def formatter():
    from backend.mod_c_explainer.formatter import ModCFormatter

    try:
        return ModCFormatter()
    except Exception as exc:  # pragma: no cover - depends on LLM config
        pytest.skip(f"ModCFormatter unavailable: {exc}")


# --------------------------------------------------------- the verdict guard


def test_explanation_does_not_move_the_verdict(grader):
    """Turning attribution on must not change a single decision.

    If this fails, explainability is not a reporting layer -- it is a second
    model, and the deployed verdict no longer matches the one being explained.
    """
    for entry in _entries():
        patient_id = entry["patient_id"]
        record = _record(patient_id)
        plain = model_client.grade_record(record, explain=False)
        explained = model_client.grade_record(record, explain=True)
        assert plain.decision_code == explained.decision_code, patient_id
        assert plain.proba == explained.proba, patient_id


# ---------------------------------------------------------- wiring is honest


def test_grade_record_attaches_real_attribution(grader):
    record = _record("IND-03-IDA-FEMALE")
    detail = model_client.grade_record(record)

    assert detail.explanation is not None
    explanation = detail.explanation
    assert explanation.explained_label == "IDA"
    assert explanation.gnn_available is True
    assert explanation.contributions, "no contributions produced"

    shares = [c.share_pct for c in explanation.contributions]
    assert sum(abs(s) for s in shares) == pytest.approx(100.0, abs=0.5)
    # ranked, so the headline biomarker is first
    assert abs(shares[0]) == max(abs(s) for s in shares)


def test_explanation_can_be_disabled(grader):
    detail = model_client.grade_record(_record("IND-03-IDA-FEMALE"), explain=False)
    assert detail.explanation is None


def test_contributions_name_real_biomarkers(grader):
    """The old field was a hardcoded literal naming `tibc` -- a marker this
    model has never seen and this report never measured."""
    detail = model_client.grade_record(_record("IND-03-IDA-FEMALE"))
    features = {c.feature for c in detail.explanation.contributions}
    assert features <= set(BASE_NAMES) | {
        "srivastava_index",
        "mentzer_index",
        "cell_hb_density",
    }
    assert "tibc" not in features


def test_imputed_values_are_marked_end_to_end(grader):
    """IND-07 is CBC-only: ferritin/B12/folate were never measured.

    An explanation leaning on a guessed ferritin must say so, or it reads as
    evidence from a test the patient never took.
    """
    detail = model_client.grade_record(_record("IND-07-CBC-ONLY-FEMALE"))
    assert detail.explanation is not None

    estimated = {
        c.feature for c in detail.explanation.contributions if c.estimated
    }
    assert {"ferritin", "b12", "folate"} <= estimated
    assert any("measured" in caveat or "estimated" in caveat
               for caveat in detail.explanation.caveats)


def test_grey_zone_explanation_carries_the_band(grader):
    """IND-04 is a textbook B12 picture the model declines to commit on."""
    detail = model_client.grade_record(_record("IND-04-MEGALOBLASTIC-B12-MALE"))
    assert detail.decision_code == 4
    assert detail.explanation is not None
    assert detail.explanation.explained_label in {
        "IDA", "B12 Deficiency", "Folate Deficiency"
    }
    assert detail.explanation.grey_zone_band == [
        pytest.approx(grader.lower_bound),
        pytest.approx(grader.upper_bound),
    ]
    assert detail.explanation.p_b12_conditional is not None


# ------------------------------------------------------- formatter behaviour


def _parameter(key: str, name: str, deviation: float) -> BloodParameter:
    return BloodParameter(
        key=key,
        name=name,
        value=10.0,
        unit="g/dL",
        normal_range="12.0 - 15.5",
        status="Low",
        deviation=deviation,
        model_input=True,
        confidence=0.9,
    )


def test_formatter_labels_model_attribution(formatter):
    explanation = ModelExplanation(
        explained_class=1,
        explained_label="IDA",
        contributions=[
            FeatureAttribution(feature="ferritin", label="Ferritin", share_pct=42.0),
            FeatureAttribution(feature="hemoglobin", label="Hemoglobin", share_pct=31.0),
            FeatureAttribution(feature="mcv", label="MCV", share_pct=8.0),
        ],
    )
    contributors = formatter._contributors("iron", explanation, [])

    assert [c.source for c in contributors] == ["model", "model", "model"]
    assert contributors[0].impact_pct == pytest.approx(42.0)
    assert contributors[0].direction == "positive"


def test_formatter_marks_imputed_contributors(formatter):
    explanation = ModelExplanation(
        explained_class=1,
        explained_label="IDA",
        contributions=[
            FeatureAttribution(
                feature="ferritin",
                label="Ferritin",
                share_pct=50.0,
                estimated=True,
            )
        ],
    )
    contributors = formatter._contributors("iron", explanation, [])
    assert contributors[0].estimated is True


def test_formatter_falls_back_and_says_so(formatter):
    """With no explanation the heuristic still runs -- but it must announce
    itself, not pass off a deviation for a model weight."""
    parameters = [
        _parameter("hemoglobin", "Hemoglobin", -0.25),
        _parameter("mcv", "MCV", -0.18),
    ]
    contributors = formatter._contributors("iron", None, parameters)

    assert contributors, "fallback produced nothing"
    assert all(c.source == "deviation" for c in contributors)
    assert contributors[0].biomarker == "Hemoglobin"
    assert contributors[0].impact_pct == pytest.approx(25.0)


def test_fallback_is_used_when_nothing_matches_the_deficiency(formatter):
    """An explanation that names no iron markers cannot explain an iron card."""
    explanation = ModelExplanation(
        explained_class=3,
        explained_label="Folate Deficiency",
        contributions=[
            FeatureAttribution(feature="folate", label="Folate", share_pct=60.0)
        ],
    )
    parameters = [_parameter("hemoglobin", "Hemoglobin", -0.25)]
    contributors = formatter._contributors("iron", explanation, parameters)

    assert all(c.source == "deviation" for c in contributors)


# ------------------------------------------------------ counterfactual wiring


@pytest.fixture(scope="module")
def explained_with_counterfactuals(grader):
    """One expensive run: the boundary search is seconds, not milliseconds."""
    return model_client.grade_record(
        _record("IND-03-IDA-FEMALE"), counterfactuals=True
    )


def test_counterfactuals_are_off_by_default(grader):
    """The search costs seconds per patient, so it must not run unasked.

    If this fails, every upload got several seconds slower and nobody chose it.
    """
    detail = model_client.grade_record(_record("IND-03-IDA-FEMALE"))
    assert detail.counterfactuals is None


def test_counterfactuals_can_be_requested(grader, explained_with_counterfactuals):
    detail = explained_with_counterfactuals
    assert detail.counterfactuals is not None
    assert detail.counterfactuals.targets, "no boundary found on a clear IDA case"
    assert detail.counterfactuals.caveats


def test_requesting_counterfactuals_does_not_move_the_verdict(
    grader, explained_with_counterfactuals
):
    """Same guarantee as attribution: the search is reporting, not inference."""
    plain = model_client.grade_record(_record("IND-03-IDA-FEMALE"))
    assert plain.decision_code == explained_with_counterfactuals.decision_code
    assert plain.proba == explained_with_counterfactuals.proba


def test_counterfactual_targets_carry_units_and_verified_boundaries(
    grader, explained_with_counterfactuals
):
    """A boundary without units is a number a clinician cannot act on safely."""
    detail = explained_with_counterfactuals
    row = {"patient_id": "IND-03-IDA-FEMALE"}
    row.update(_record("IND-03-IDA-FEMALE")["model_input"])

    for target in detail.counterfactuals.targets:
        assert target.unit, f"{target.feature} reported no unit"
        assert target.to_etiology != detail.etiology

        probe = dict(row)
        probe[target.feature] = target.target_value
        landed = int(grader.predict_proba([probe])["decision"][0])
        assert landed == DECISION_NAMES.index(target.to_etiology), (
            f"{target.feature} → {target.target_value} should reach "
            f"{target.to_etiology}, model says {DECISION_NAMES[landed]}"
        )


