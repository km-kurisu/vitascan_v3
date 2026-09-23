"""composer — merge multiple per-file `pdf_raw` extractions into one composite \
report.

A single patient often uploads several documents (CBC/hemogram PDF, iron-studies
scan, B12/folate page). Each file produces its own `pdf_raw` artifact with
partial `biomarkers`. This module composes them into a single pdf_raw-shaped
dict that `GraderInputBuilder.build(...)` can consume as if it were one report:

  * `biomarkers`  — key-level first-file-wins union across all parts. Every
    alias key survives, so the normalizer's alias resolution can still pick the
    best descriptive slug (e.g. `total_rbc_count` from one file over a junk
    `rbc` from another).
  * `raw_text`    — concatenation of every part's raw text with file headers,
    so the raw-text fallback scan can reach values that only exist in a
    specific page.
  * `composite_parts` + `biomarker_provenance` — breadcrumbs for the UI /
    audit trail, recording which file every merged value came from.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

COMPOSITE_FILE_HEADER = "===== FILE: {filename} (part {i}/{n}) ====="


def compose_pdf_raw(
    docs: List[Dict[str, Any]],
    patient_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge one-or-more pdf_raw extraction dicts into a single composite pdf_raw.

    Ordering is significant: for every alias *key* the first file that has it
    wins (later duplicates are ignored). Empty biomarker entries and non-dict
    values are skipped. `raw_text` is joined with the file headers below.
    """
    if not docs:
        raise ValueError("compose_pdf_raw requires at least one pdf_raw document")

    raw_text_chunks: List[str] = []
    merged_biomarkers: Dict[str, Any] = {}
    provenance: List[Dict[str, Any]] = []
    parts: List[Dict[str, Any]] = []

    for i, doc in enumerate(docs, start=1):
        if not isinstance(doc, dict):
            continue
        filename = doc.get("filename") or f"part{i}"
        part_pid = doc.get("patient_id")
        text = doc.get("raw_text") or ""

        raw_text_chunks.append(
            f"{COMPOSITE_FILE_HEADER.format(filename=filename, i=i, n=len(docs))}\n{text}"
        )

        markers = doc.get("biomarkers") or {}
        part_count = sum(1 for v in markers.values() if isinstance(v, dict))
        for key, entry in markers.items():
            if not isinstance(entry, dict):
                continue
            if key in merged_biomarkers:
                continue
            merged_biomarkers[key] = entry
            provenance.append({
                "key": key,
                "filename": filename,
                "patient_id": part_pid,
                "value": entry.get("value", entry.get("raw_value")),
                "unit": entry.get("unit"),
                "ref_range": entry.get("ref_range"),
            })

        parts.append({
            "filename": filename,
            "patient_id": part_pid,
            "json_storage_path": doc.get("json_storage_path"),
            "biomarker_count": part_count,
        })

    resolved_patient = patient_id or docs[0].get("patient_id") or "PAT-UNKNOWN"

    composite: Dict[str, Any] = {
        "patient_id": resolved_patient,
        "filename": f"composite/{resolved_patient} ({len(parts)} parts)",
        "composite_parts": parts,
        "biomarker_provenance": provenance,
        "biomarkers": merged_biomarkers,
        "raw_text": "\n\n".join(raw_text_chunks),
    }

    return composite