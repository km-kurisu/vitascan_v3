"""Tests for Level 2 (counterfactual) explanations.

The load-bearing test is :func:`test_reported_flips_actually_flip`. A
counterfactual is a claim about what the model would decide on a row the
patient did not have, and the only thing that makes such a claim safe is
re-running the model on that row. If that test fails, every number this module
reports is a guess dressed as a boundary.

These tests are slow -- the GNN cannot be batched (see the module docstring),
so each reported boundary costs a grid sweep plus a bisection. Results for the
shared patients are computed once per session by fixtures.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from backend.path_b_blood_report.mod_b3_grader.explainability.counterfactual import (
    DEFAULT_TOL_FRAC,
    PLAUSIBLE_RANGES,
    UNIVARIATE_CAVEAT,
    counterfactual_flips,
)
from backend.path_b_blood_report.mod_b3_grader.grader_model.features import (
    BASE_NAMES,
)

MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "shared"
    / "sample_reports"
    / "test_pdfs"
    / "MANIFEST.json"
)

#: Small grid: these tests are about correctness of the boundary, and bisection
#: tightens whatever the grid brackets. A coarse grid also exercises the
#: "hit at the grid edge" path more often than a fine one would.
SAMPLES = 9

#: IND-07 is CBC-only; the pipeline imputes these.
IMPUTED_IN_IND07 = {"ferritin", "b12", "folate"}


def _manifest_entries():
    with MANIFEST.open(encoding="utf-8") as handle:
        return json.load(handle)


def _entry(patient_id: str) -> dict:
    for candidate in _manifest_entries():
        if candidate["patient_id"] == patient_id:
            return candidate
    raise AssertionError(f"{patient_id} missing from MANIFEST.json")


def _row_for(entry) -> dict:
    """Model row from a manifest entry. Missing values arrive as NaN."""
    biochem = entry["biochemistry"]
    return {
        "patient_id": entry["patient_id"],
        **{
            name: (np.nan if biochem.get(name) is None else biochem.get(name))
            for name in BASE_NAMES
        },
    }


@pytest.fixture(scope="module")
def grader():
    from backend.path_b_blood_report.mod_b3_grader.grader_model.predict import (
        AnemiaGrader,
    )

    return AnemiaGrader()


@pytest.fixture(scope="module")
def cf_ida(grader):
    return counterfactual_flips(
        grader, _row_for(_entry("IND-03-IDA-FEMALE")), samples=SAMPLES
    )


@pytest.fixture(scope="module")
def cf_healthy(grader):
    return counterfactual_flips(
        grader, _row_for(_entry("IND-01-HEALTHY-MALE")), samples=SAMPLES
    )


@pytest.fixture(scope="module")
def cf_cbc_only(grader):
    return counterfactual_flips(
        grader,
        _row_for(_entry("IND-07-CBC-ONLY-FEMALE")),
        imputed_features=IMPUTED_IN_IND07,
        samples=SAMPLES,
    )


@pytest.fixture(scope="module")
def cf_grey(grader):
    return counterfactual_flips(
        grader, _row_for(_entry("IND-06-MIXED-GREYZONE-MALE")), samples=SAMPLES
    )


# ------------------------------------------------------------------ the guard


def test_reported_flips_actually_flip(grader, cf_ida, cf_healthy, cf_grey):
    """Every boundary reported must survive being re-run on the deployed model.

    This is the whole contract. A counterfactual that the model does not
    reproduce is worse than no counterfactual, because it reads as a measured
    clinical target.
    """
    checked = 0
    for report, entry in (
        (cf_ida, _entry("IND-03-IDA-FEMALE")),
        (cf_healthy, _entry("IND-01-HEALTHY-MALE")),
        (cf_grey, _entry("IND-06-MIXED-GREYZONE-MALE")),
    ):
        row = _row_for(entry)
        for flip in report.flips:
            probe = dict(row)
            probe[flip.feature] = flip.target_value
            landed = int(grader.predict_proba([probe])["decision"][0])
            assert landed == flip.to_decision, (
                f"{entry['patient_id']}: {flip.feature} at {flip.target_value} "
                f"should reach decision {flip.to_decision}, model says {landed}"
            )
            assert flip.verified is True
            checked += 1

    assert checked, "no flips were produced by any corpus patient -- suspicious"


def test_the_boundary_is_tight(grader, cf_ida):
    """The reported value should be the crossing, not the coarse grid point.

    Stepping back toward the patient's own value by a comfortable margin must
    land on the original verdict again; if it does not, bisection stopped early
    and the reported target is further away than it needs to be.
    """
    flip = next((f for f in cf_ida.flips if not f.at_range_boundary), None)
    if flip is None:
        pytest.skip("no interior boundary on this patient")

    entry = _entry("IND-03-IDA-FEMALE")
    row = _row_for(entry)
    span = abs(PLAUSIBLE_RANGES[flip.feature][1] - PLAUSIBLE_RANGES[flip.feature][0])
    margin = 5.0 * DEFAULT_TOL_FRAC * span
    step_back = flip.target_value - np.sign(flip.delta) * margin

    probe = dict(row)
    probe[flip.feature] = float(step_back)
    landed = int(grader.predict_proba([probe])["decision"][0])
    assert landed == cf_ida.decision_code, (
        f"{flip.feature}={step_back} is still past the boundary; the reported "
        f"target {flip.target_value} is wider than the true crossing"
    )


# ------------------------------------------------------------- what it refuses


def test_imputed_features_are_refused_not_searched(cf_cbc_only):
    """IND-07 has no ferritin/B12/folate to have a counterfactual about.

    Searching them would produce a confident number for a test the patient
    never took, which is exactly the failure this module exists to avoid.
    """
    assert set(cf_cbc_only.refused) == IMPUTED_IN_IND07
    searched = {f.feature for f in cf_cbc_only.flips}
    assert not (searched & IMPUTED_IN_IND07)
    assert any("estimated rather than measured" in c for c in cf_cbc_only.caveats)


def test_every_feature_is_accounted_for(cf_ida, cf_cbc_only, cf_grey):
    """Reported, unreachable and refused must partition the nine base markers.

    A feature that silently vanishes from all three lists is a bug that would
    read on screen as "the model had nothing to say", which is a different and
    much stronger claim than "we did not look".
    """
    for report in (cf_ida, cf_cbc_only, cf_grey):
        buckets = [
            {f.feature for f in report.flips},
            set(report.unreachable),
            set(report.refused),
        ]
        union = set().union(*buckets)
        assert union == set(BASE_NAMES), (
            f"unaccounted: {set(BASE_NAMES) - union}"
        )
        assert sum(len(b) for b in buckets) == len(union), "a feature is in two buckets"


# ------------------------------------------------------------------ behaviour


def test_verdict_matches_the_deployed_model(grader, cf_ida, cf_healthy, cf_grey):
    for report, patient_id in (
        (cf_ida, "IND-03-IDA-FEMALE"),
        (cf_healthy, "IND-01-HEALTHY-MALE"),
        (cf_grey, "IND-06-MIXED-GREYZONE-MALE"),
    ):
        row = _row_for(_entry(patient_id))
        live = int(grader.predict_proba([row])["decision"][0])
        assert report.decision_code == live


def test_a_reported_flip_never_returns_the_same_verdict(cf_ida, cf_healthy, cf_grey):
    for report in (cf_ida, cf_healthy, cf_grey):
        for flip in report.flips:
            assert flip.to_decision != report.decision_code
            assert flip.target_value != flip.current_value
            assert flip.direction in {"increase", "decrease"}
            assert (flip.direction == "increase") == (flip.delta > 0)


def test_flips_are_ordered_nearest_first(cf_ida, cf_grey):
    for report in (cf_ida, cf_grey):
        deltas = [abs(f.delta) for f in report.flips]
        assert deltas == sorted(deltas)
        if report.flips:
            assert report.nearest() is report.flips[0]


def test_healthy_patient_boundary_leads_to_a_deficiency(cf_healthy):
    """IND-01 is healthy; any boundary it has should leave the healthy class."""
    assert cf_healthy.decision_code == 0
    for flip in cf_healthy.flips:
        assert flip.to_decision != 0


def test_target_decision_is_respected(grader):
    """Restricting the search must not report a route to a different class."""
    report = counterfactual_flips(
        grader,
        _row_for(_entry("IND-03-IDA-FEMALE")),
        target_decision=0,
        samples=SAMPLES,
    )
    assert report.target_decision == 0
    for flip in report.flips:
        assert flip.to_decision == 0
        probe = _row_for(_entry("IND-03-IDA-FEMALE"))
        probe[flip.feature] = flip.target_value
        assert int(grader.predict_proba([probe])["decision"][0]) == 0
    assert any("restricted to" in c for c in report.caveats)


def test_the_limit_of_the_method_is_always_disclosed(cf_ida, cf_healthy, cf_cbc_only):
    """Univariate counterfactuals are a model statement, not clinical advice.

    The caveat is the only thing standing between these numbers and someone
    reading them as "eat more iron and you will be fine", so it is not optional.
    """
    for report in (cf_ida, cf_healthy, cf_cbc_only):
        assert UNIVARIATE_CAVEAT in report.caveats
        assert any("hand-set clinical ranges" in c for c in report.caveats)


def test_search_does_not_mutate_the_row(grader):
    """The caller's record must come back untouched."""
    row = _row_for(_entry("IND-03-IDA-FEMALE"))
    before = copy.deepcopy(row)
    counterfactual_flips(grader, row, samples=SAMPLES)
    for name in BASE_NAMES:
        assert (row[name] == before[name]) or (
            np.isnan(row[name]) and np.isnan(before[name])
        ), f"{name} was modified by the search"
