"""Process all archived pdf_raw extractions into model-usable grader inputs.

For every `<patient>_latest.json` in shared/extractions/pdf_raw it:
  1. Runs GraderInputBuilder (clean -> resolve the 9 model biomarkers -> missing=null).
  2. Saves a per-patient artifact to shared/extractions/grader_input/.
  3. Appends its `to_payload()` record to a batch payload.

The batch payload is a plain JSON *list* of patient dicts of the exact shape
`vitascan_grader_model.AnemiaGrader.grade(...)` expects — the file is written to
`shared/extractions/grader_input/batch_payload_latest.json` (plus a timestamped copy).

Usage:
    python -m backend.path_b_blood_report.mod_b2_normalizer.run_grader_input_pipeline
"""
from __future__ import annotations

import json
import time
import argparse
from datetime import datetime
from pathlib import Path

from backend.path_b_blood_report.mod_b2_normalizer.grader_input import GraderInputBuilder
from backend.shared.storage import SUBDIRS, save_json_artifact

PDF_RAW_DIR = SUBDIRS["pdf_raw"]
GRADER_INPUT_DIR = SUBDIRS["grader_input"]


def process_pdf_raw_dir(pdf_raw_dir: Path, builder: GraderInputBuilder, save: bool = True):
    records = []
    for f in sorted(pdf_raw_dir.glob("*_latest.json")):
        try:
            pdf_raw = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"SKIP {f.name}: bad json ({e})")
            continue
        record = builder.build(pdf_raw)
        records.append(record)
        if save:
            save_json_artifact("grader_input", record["patient_id"], record)
        print(f"[{record['na_count']}/9 NaN] {record['patient_id']:<20} "
              f"{record['model_input']}")
    return records


def save_batch_payload(records, grader_input_dir: Path) -> Path:
    rows = [{"patient_id": r["patient_id"], **r["model_input"]} for r in records]
    latest = grader_input_dir / "batch_payload_latest.json"
    ts = grader_input_dir / f"batch_payload_{int(time.time())}.json"
    for p in (latest, ts):
        p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return latest


def main():
    ap = argparse.ArgumentParser(description="pdf_raw -> model-usable grader inputs")
    ap.add_argument("--no-save", action="store_true", help="dry run (build only)")
    ap.add_argument("--pdf-dir", type=Path, default=PDF_RAW_DIR)
    args = ap.parse_args()

    builder = GraderInputBuilder()
    print(f"imputer loaded: {builder.imputer is not None}")
    print(f"scanning {args.pdf_dir}\n")
    records = process_pdf_raw_dir(args.pdf_dir, builder, save=not args.no_save)

    if args.no_save:
        return
    latest = save_batch_payload(records, GRADER_INPUT_DIR)
    covered = sum(9 - r["na_count"] for r in records)
    print(f"\nSaved {len(records)} grader_input artifacts + batch payload -> {latest}")
    print(f"Total biomarker values resolved: {covered} / {len(records) * 9}")
    print(f"Generated: {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()