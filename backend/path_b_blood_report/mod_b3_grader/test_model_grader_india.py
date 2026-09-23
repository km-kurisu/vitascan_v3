"""Integration test: the copied deployed AnemiaGrader (mod_b3_grader/grader_model)
running inside the Vitascan V3 flow on the 6 synthetic India-compliant composite
reports (backend/shared/sample_reports/IND-*).

Flow under test (exactly the production chain):
    sample pdf_raw  ->  GraderInputBuilder.build()          (normalizer record)
                    ->  PathBGrader.grade(..., grader_input=record)
                                      (legacy severity + AnemiaGrader verdict)

The graded results + a markdown report are logged inside vitascan_v3
(`backend/shared/extractions/mod_b3_grader/INDIA_COMPOSITE_MODEL_TEST.md`) and as
per-patient `model_grade` artifacts.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from backend.path_b_blood_report.mod_b2_normalizer.grader_input import GraderInputBuilder
from backend.path_b_blood_report.mod_b3_grader.grader import PathBGrader
from backend.shared.storage import save_json_artifact

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "shared" / "sample_reports"
REPORT_DIR = Path(__file__).resolve().parents[2] / "shared" / "extractions" / "mod_b3_grader"
REPORT_PATH = REPORT_DIR / "INDIA_COMPOSITE_MODEL_TEST.md"

# Intended diagnosis per synthetic composite and the decision_code we expect.
INTENDED = {
    "IND-01-HEALTHY-MALE": ("No Anemia", 0),
    "IND-02-HEALTHY-FEMALE": ("No Anemia", 0),
    "IND-03-IDA-FEMALE": ("IDA", 1),
    "IND-04-MEGALOBLASTIC-B12-MALE": ("B12 Deficiency", 4),  # severe/low -> triage band
    "IND-05-MEGALOBLASTIC-FOLATE-FEMALE": ("Folate Deficiency", 3),
    "IND-06-MIXED-GREYZONE-MALE": ("Grey Zone Triage", 4),
}

DECISION_NAMES = ["No Anemia", "IDA", "B12 Deficiency", "Folate Deficiency", "Grey Zone Triage"]


def load_samples() -> list:
    samples = []
    for p in sorted(SAMPLE_DIR.glob("IND-*_latest.json")):
        samples.append((p.stem.removesuffix("_latest"), json.loads(p.read_text())))
    return samples


def match_level(pid: str, decision_code: int, proba: dict) -> str:
    intended_name, intended_code = INTENDED[pid]
    if decision_code == intended_code:
        return "Y"
    if decision_code == 4 and max(proba, key=proba.get) == intended_name:
        return "P"
    return "N"


def build_report(results: list, built: dict) -> str:
    L = []
    A = L.append
    A("# AnemiaGrader (mod_b3_grader) × India composite reports — Vitascan V3 test report")
    A("")
    A(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    A("- Flow: `sample pdf_raw -> GraderInputBuilder.build() -> "
      "PathBGrader.grade(grader_input=...)`")
    A(f"- Model: xgb cascade + hetero-GNN hybrid (blend alpha per decision)")
    A(f"- Samples: `backend/shared/sample_reports/IND-*_latest.json` ({len(results)} reports)")
    A("")
    A("## 1. Verdicts")
    A("")
    A("| Report | Intended | decision_code | Etiology | P(top) | Match |")
    A("|--------|----------|---------------|----------|--------|-------|")
    for pid, r in results:
        top = max(r["model"]["proba"], key=r["model"]["proba"].get)
        A(f"| `{pid}` | {INTENDED[pid][0]} | {r['model']['decision_code']} | "
          f"**{r['model']['etiology']}** | {top} ({r['model']['proba'][top]:.2f}) | "
          f"{match_level(pid, r['model']['decision_code'], r['model']['proba'])} |")
    A("")
    A("`4` = Grey Zone Triage. Match: `Y` exact, `P` borderline (intended definite "
      "class, but the model sent it to triage with that class on top).")
    A("")
    A("## 2. Per-report details")
    A("")
    for pid, r in results:
        A(f"### {pid}")
        A("")
        A("| Field | Value |")
        A("|-------|-------|")
        A(f"| model_input | `{json.dumps(r['model_input'])}` |")
        A(f"| na_count | {r['model']['na_count']} |")
        A(f"| complete_input | {r['model']['complete_input']} |")
        A(f"| decision_code | {r['model']['decision_code']} ({r['model']['etiology']}) |")
        A(f"| P blended | `{json.dumps(r['model']['proba'])}` |")
        A(f"| P xgb | `{json.dumps(r['model']['proba_xgb'])}` |")
        A(f"| P gnn | `{json.dumps(r['model']['proba_gnn'])}` |")
        A(f"| alpha | `{r['model']['alpha']}` |")
        A(f"| model_confidence | `{r['model_confidence']}` |")
        A(f"| severity overlay | `{json.dumps(r['severity_overlay'])}` |")
        A("")
        if r["model"].get("imputed_features"):
            A("Imputed 12-feature row:")
            A("")
            A("```")
            for k, v in r["model"]["imputed_features"].items():
                A(f"  {k:<22} {v if v is None else round(v, 4)}")
            A("```")
            A("")
    A("## 3. Notes")
    A("")
    A("- Reports are synthetic composites (no real patient data), used to prove the "
      "copied model is wired into the Vitascan V3 flow end-to-end.")
    A("- `IND-04` (severe B12 deficiency) is routed to Grey Zone Triage by design; "
      "`P(B12 Deficiency)` remains the top blended class.")
    A("")
    return "\n".join(L) + "\n"


def test_india_composites_through_full_flow():
    builder = GraderInputBuilder()
    grader = PathBGrader()

    results = []
    built = {}
    for pid, pdf_raw in load_samples():
        record = builder.build(pdf_raw)
        output = grader.grade(pid, {}, grader_input=record)
        assert output.model is not None, f"{pid}: model grade missing"
        assert output.model.na_count == 0, f"{pid}: expected 9/9 biomarkers, got {output.model.na_count} NaN"
        detail = output.model
        save_json_artifact("model_grade", pid, detail.model_dump())
        results.append((pid, {
            "model_input": record["model_input"],
            "model": detail.model_dump(),
            "model_confidence": output.model_confidence,
            "severity_overlay": {k: v.model_dump() for k, v in output.severity.items()
                                 if v.model == "anemia_grader"},
        }))
        built[pid] = output
        print(f"[{detail.na_count}/9 NaN] {pid:<34} code={detail.decision_code} "
              f"etiology={detail.etiology} topp={max(detail.proba.values()):.3f}")

    # match every report against its intended diagnosis
    for pid, r in results:
        assert match_level(pid, r["model"]["decision_code"], r["model"]["proba"]) in ("Y", "P"), \
            f"{pid}: model did not land near intended {INTENDED[pid]} -> {r['model']['etiology']}"
    # IND-04: B12 must be the top blended class even though it gets triaged
    for pid, r in results:
        if pid == "IND-04-MEGALOBLASTIC-B12-MALE":
            assert max(r["model"]["proba"], key=r["model"]["proba"].get) == "B12 Deficiency"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(results, built))
    print(f"\nLogged report -> {REPORT_PATH}")
    assert REPORT_PATH.exists()