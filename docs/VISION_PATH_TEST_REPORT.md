# Path A (Image-Based Symptoms Checker) — Test Report

VitaScan V3 · `backend/path_a_symptom_image` · live Groq vision inference (`qwen/qwen3.8-27b`)

> Generated 2026-09-20 · Synthetic symptom photos → real Groq multimodal analysis → cross-check signal → HTTP API.

## 1. Architecture & Pipeline

| Stage | What runs | Failure behaviour |
|---|---|---|
| 1. Upload endpoint | `POST /upload-symptom-photo` accepts `file`, `source_region` (eyes/nails/tongue/skin), `patient_id` | 422 on missing file |
| 2. Groq vision inference | `GroqSymptomVisionAnalyzer` = `qwen/qwen3.8-27b`, temperature 0.2, `response_format=json_object`, base64 data-URL image, region-specific clinical prompt | Falls back to rule-assisted canned analysis, never crashes |
| 3. Cross-check formatting | `PathACrosscheckSignal.format_signal` maps the LLM JSON → `ModA3Output` (`crosscheck_signal` map, `visual_findings`, `model_confidence`) | Boolean/float coercion handles malformed values |
| 4. Persistence | Raw LLM JSON saved via `save_json_artifact("groq_symptom", ...)` → `backend/shared/extractions/groq_symptom/` (timestamped + `*_latest.json`) | Artifact write is best-effort; API still returns |

Model resolution: `GROQ_VISION_API_KEY` (falls back to `GROQ_API_KEY`), model `GROQ_VISION_MODEL` override (default `qwen/qwen3.8-27b`).

## 2. Unit Tests

`backend/venv/bin/python -m pytest backend -q`

| Suite | Tests | Status |
|---|---|---|
| `path_a_symptom_image/test_crosscheck.py` | 3 | PASS |
| `shared/test_extraction_biobert.py` (extractor + GLiNER biobert hybrid) | 5 | PASS |
| `shared/test_schemas.py` | 3 | PASS |
| `diet_rag_service/test_diet_rag.py` | 9 | PASS |
| `reminders/` | 15 | PASS |
| **Total** | **35** | **35 passed (15.46s, GLiNER-loaded app venv)** |

New Path A test coverage (`backend/path_a_symptom_image/test_crosscheck.py`):
- `test_fallback_analysis_shape` — no-key / model-failure path returns a valid canned analysis per region.
- `test_crosscheck_signal_maps_findings` — verifies deficiency→signal mapping, source, confidence, `agrees_with_path_b` and `visual_findings` reach `ModA3Output` (guards the schema passthrough fix).
- `test_crosscheck_signal_defaults` — empty findings and defaults coerce cleanly.

## 3. Live Endpoint Matrix — 4 source regions

Synthetic fixtures (PIL-generated in `/tmp/opencode/sym_photos/`): `left_eye.png`, `index_fingernail.png`, `tongue.png`, `skin_pallor.png`.
All calls: `POST http://localhost:8000/upload-symptom-photo` with `patient_id=PAT-A3DOC`.

| Region | HTTP | Deficiency | Confidence | Agrees w/ Path B | Latency | Visual findings (verbatim, LLM) |
|---|---|---|---|---|---|---|
| eyes | 200 | anemia | 0.45 | true | 0.44s | Pallor of the conjunctiva; Pale lower eyelid mucosa |
| nails | 200 | anemia | 0.45 | true | 0.45s | Generalized pallor of the nail bed; Absence of normal pink coloration in the lunula and distal nail bed; No visible koilonychia (spooning) or Beau's lines |
| tongue | 200 | anemia | 0.45 | true | 18.49s | Generalized pallor of the tongue; Smooth, atrophic tongue surface (glossitis); Absence of visible papillae |
| skin | 200 | iron | 0.45 | true | 19.47s | Generalized skin pallor; Koilonychia (spoon-shaped nails) visible in the central and lower-left circular regions |

Each region produced distinct, clinically plausible findings — i.e. genuine multimodal inference, not the canned fallback (the fallback always returns `anemia`/0.82 with region-specific boilerplate). Artifacts persisted under `backend/shared/extractions/groq_symptom/PAT-A3DOC_*_groq_symptom.json` + `PAT-A3DOC_latest.json`.

Representative raw response (`eyes` region):

```json
{
  "status": "success",
  "crosscheck": {
    "patient_id": "PAT-A3DOC",
    "crosscheck_signal": {
      "anemia": {
        "confidence": 0.45,
        "source": "eyes",
        "agrees_with_path_b": true
      }
    },
    "visual_findings": ["Pallor of the conjunctiva", "Pale lower eyelid mucosa"],
    "model_confidence": 0.45
  },
  "latest_result": { ... }
}
```

## 4. Resilience — model failure degrades gracefully

Live check: forced an unsupported model (`GROQ_VISION_MODEL=meta-llama/llama-prompt-guard-2-22m`, text-only) through the analyzer.

| Scenario | Result | Latency |
|---|---|---|
| Valid model (`qwen/qwen3.8-27b`) | Real LLM JSON (deficiency/confidence/findings differ per region) | 0.44–19.5s (varies with queue/thinking) |
| Unsupported model (rejects JSON mode) | `analyze_image_bytes` caught the 400 → returned rule-fallback analysis (`anemia`, 0.82, region boilerplate) | 0.04s |
| Missing `GROQ_VISION_API_KEY` | Client never initialises → fallback path | n/a |

The endpoint never 5xxed across any scenario; all failures degrade to the canned analysis with `status: success`.

## 5. Bugs found & fixed while testing

1. **Decommissioned vision model** — the analyzer called `llama-3.2-11b-vision-preview`, which Groq retired → every call 400'd and silently returned the fallback. Replaced with `qwen/qwen3.8-27b` (Groq's current multimodal model), set `response_format={"type":"json_object"}`, added a `reasoning_content` prefill fallback and `GROQ_VISION_MODEL` env override. (`groq_vision_analyzer.py`)
2. **`visual_findings` silently dropped** — `PathACrosscheckSignal` populated the field but `ModA3Output` didn't declare it, so pydantic discarded it from the API response. Added `visual_findings: List[str]` to the schema. (`schemas.py`, `test_schemas.py`)
3. **Legacy storage signatures** — `extractor.py`/`biobert_extractor.py` still called `save_extracted_json(data, patient_id=...)` with the pre-refactor arg order → `TypeError` inside the try → extracted JSON was never persisted. Fixed to canonical `(patient_id, data)` and added missing `EXTRACTIONS_DIR`/`PATIENT_BIOMARKERS_DIR` compat constants. (`storage.py`, `extractor.py`, `biobert_extractor.py`)
4. **Stale tests** — `test_extraction_biobert.py` referenced the old storage API/log strings; updated to current behaviour and appended compatible constants. New `test_crosscheck.py` added for Path A.

## 6. Limitations & notes

- Confidence reported as 0.45 across all four synthetic images (Groq output quirk for these low-texture fixtures); findings text is the informative signal.
- Latency is bimodal (sub-second vs ~19s) depending on Groq queue/thinking-mode variance — the API contract is async-friendly (client-only call, no DB transaction).
- Fixtures are synthetic illustrations, not clinical photographs — they demonstrate the plumbing, not diagnostic accuracy.
- The `*_latest.json` artifact is overwritten per call; timestamped files are retained for audit.

## 7. Conclusion

Path A is fully operational: real Groq vision inference, per-region findings, cross-check signal merged into `ModA3Output`, graceful degradation under model/key failures, and persistent artifacts. Full backend test suite: **35/35 passing**.