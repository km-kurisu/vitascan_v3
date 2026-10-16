# VitaScan V3

**An explainable AI system for nutritional-deficiency anaemia screening from blood reports, built for Indian clinical reference ranges.**

VitaScan reads a routine blood report (PDF or scanned image), extracts the biomarkers that
matter for an anaemia workup, runs a hybrid machine-learning grader over them, and returns a
deficiency verdict with an explanation of *why* — which biomarkers drove it, and what would
have to change for the verdict to move.

It is built around the three deficiency anaemias that dominate in India — iron deficiency
(IDA), vitamin B12 deficiency, and folate deficiency — using Indian adult reference intervals
and the ICMR-NIN 2020 recommended dietary allowances rather than Western defaults.

---

## Table of Contents

- [The problem](#the-problem)
- [What it does](#what-it-does)
- [Architecture](#architecture)
- [The anaemia grading model](#the-anaemia-grading-model)
- [Explainability (XAI)](#explainability-xai)
- [Path A — symptom photo cross-check](#path-a--symptom-photo-cross-check)
- [Diet recommendations (FSSAI / IFCT2017 RAG)](#diet-recommendations-fssai--ifct2017-rag)
- [Reminders](#reminders)
- [Tech stack](#tech-stack)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Trying it end to end](#trying-it-end-to-end)
- [API reference](#api-reference)
- [Testing](#testing)
- [Design decisions and known limits](#design-decisions-and-known-limits)
- [Documentation](#documentation)
- [Disclaimer](#disclaimer)

---

## The problem

Anaemia in India is usually not one disease. Iron deficiency, B12 deficiency, and folate
deficiency produce overlapping symptoms and overlapping blood pictures, and a plain
haemoglobin reading cannot tell them apart — yet the treatment for each is different, and
treating the wrong one wastes months.

Distinguishing them requires reasoning over several correlated indices at once: MCV, RDW,
MCH, MCHC, ferritin, B12, folate, and derived red-cell indices such as the Mentzer and
Srivastava indices. That is exactly the kind of joint reasoning a model can do and a busy
clinician rarely has time to do by hand.

Two things make such a model hard to trust in practice:

1. **The verdict is opaque.** A probability with no reason attached is not actionable.
2. **Missing data is the norm.** Many Indian lab reports omit ferritin, B12, or folate
   entirely — a plain CBC is often all there is.

VitaScan addresses both: it imputes what is missing rather than refusing to grade, and it
reports the reasoning behind every verdict instead of only the number.

---

## What it does

| Capability | Detail |
| --- | --- |
| **Blood report ingestion** | Text-layer PDFs read directly with PyMuPDF; scanned/image-only reports fall back to Tesseract OCR with image preprocessing |
| **Biomarker extraction** | GLiNER biomedical NER (`Ihor/gliner-biomed-small-v1.0`) with BioBERT and regex fallbacks, plus a line-oriented numeric scanner |
| **Missing-value handling** | A KNN imputer fills absent biomarkers; the model always grades, and the UI states how many values were estimated |
| **Hybrid grading** | XGBoost cascade + heterogeneous GNN, probability-blended, resolved into one of five decisions |
| **Explainability** | Exact TreeSHAP over the cascade + Integrated Gradients over the GNN, blended as normalised shares |
| **Counterfactuals** | For each measured biomarker, the nearest value at which the verdict changes |
| **Symptom cross-check** | Vision-LLM analysis of a symptom photograph (pallor, koilonychia, glossitis) to corroborate or contradict the blood verdict |
| **Diet guidance** | A curated IFCT2017/ICMR-NIN food knowledge graph with a 7-day meal planner, allergy and disorder filtering |
| **Reminders** | Supplement/appointment reminders by email, with Google Calendar links and `.ics` export |
| **Localisation** | English, Hindi (हिन्दी), and Marathi (मराठी) |

---

## Architecture

VitaScan is organised as a pipeline of numbered modules. Data flows left to right and each
stage persists a JSON artifact under `backend/shared/extractions/`, so any run can be
inspected and diffed afterwards.

```
                       ┌──────────────────────────────────────────────┐
   blood report PDF ──►│ Path B — Blood report                        │
                       │                                              │
                       │  B1  extractor.py       text layer / OCR     │
                       │      biobert_extractor.py   NER → biomarkers │
                       │  B2  normalizer.py      units + deviation    │
                       │      grader_input.py    9-key model payload  │
                       │  B3  grader.py          ────┐                │
                       └─────────────────────────────┼────────────────┘
                                                     ▼
                       ┌──────────────────────────────────────────────┐
                       │ AnemiaGrader (hybrid)                        │
                       │   9 base values → 12 features                │
                       │   KNNImputer → StandardScaler                │
                       │   XGB cascade ─┐                             │
                       │   Hetero-GNN  ─┴─ p = α·p_xgb+(1−α)·p_gnn     │
                       │   triage → decision 0–4                      │
                       └──────────────────┬───────────────────────────┘
                                          ▼
                       ┌──────────────────────────────────────────────┐
   symptom photo ─────►│ Path A — cross-check (Groq vision)           │
                       └──────────────────┬───────────────────────────┘
                                          ▼
                       ┌──────────────────────────────────────────────┐
                       │ Mod C — explainer & formatter                │
                       │   verdict → deficiency cards, contributors,  │
                       │   FSSAI-checked diet advice, blood parameters│
                       └──────────────────┬───────────────────────────┘
                                          ▼
                       ┌──────────────────────────────────────────────┐
                       │ FastAPI  ──►  Next.js 14 frontend            │
                       │ Diet RAG service, reminders                  │
                       └──────────────────────────────────────────────┘
```

**Path B is the core of the project.** Path A corroborates it; Mod C explains it.

---

## The anaemia grading model

The grader is the centrepiece. It is a two-branch hybrid whose outputs are combined by a
soft vote.

### Inputs — 9 base values → 12 features

The model consumes nine biomarkers, which map directly onto what a CBC plus an anaemia panel
reports:

```
hemoglobin · mcv · rdw · ferritin · b12 · folate · rbc · mch · mchc
```

Three clinical indices are derived from them, giving the 12 features the models were trained
on:

| Feature | Formula | What it discriminates |
| --- | --- | --- |
| `srivastava_index` | `mch / rbc` | Thalassaemia trait vs iron deficiency |
| `mentzer_index` | `mcv / rbc` | Thalassaemia trait vs iron deficiency |
| `cell_hb_density` | `mch × mchc / 100` | Hypochromia severity |

Missing values are filled by a persisted KNN imputer before anything else runs, so the model
grades partial reports too.

### Branch 1 — XGBoost hierarchical cascade

Rather than one 4-class classifier, three XGBoost stages decompose the question the way a
clinician does, and the joint distribution is assembled from the conditional probabilities:

```
stage1   healthy  vs  any anaemia              → P(anaemia)
stage2a  IDA/B12  vs  folate                   → P(folate | anaemia)
stage2b  IDA      vs  B12   (isotonic-calibrated) → P(B12 | IDA or B12)
```

```
P(No Anemia) = 1 − P(anaemia)
P(IDA)       = P(anaemia) · (1 − P(folate|anaemia)) · (1 − P(B12|diff))
P(B12)       = P(anaemia) · (1 − P(folate|anaemia)) · P(B12|diff)
P(Folate)    = P(anaemia) · P(folate|anaemia)
```

Stage 2b is probability-calibrated (isotonic regression) because its output feeds the
grey-zone rule below, where a miscalibrated probability would change the decision.

### Branch 2 — heterogeneous graph neural network

A `HeteroClinicalGNN` treats each patient as a node connected to the twelve biomarker nodes
and four etiology nodes, so the model can learn relationships *between* markers rather than
only within a feature row:

- Node types: `patient`, `biomarker` (12), `etiology` (4)
- Relations: `measures` / `rev_measures` (patient ↔ biomarker) and `indicates` /
  `rev_indicates` (biomarker ↔ etiology, from a clinical knowledge edge list)
- 2 × `HeteroConv(SAGEConv, aggr="mean")`, hidden dimension 64, ELU activations
- Classifier MLP: `64 → ReLU → Dropout(0.2) → 4`

### The blend and the decision

```
p = α · p_xgb + (1 − α) · p_gnn
```

`α` and the grey-zone band bounds are read from the saved `pipeline_config.joblib` — never
hardcoded — so the reported blend weight always matches the one actually applied.

The blended distribution is then resolved by a triage cascade:

| # | Decision | Rule |
| --- | --- | --- |
| 0 | No Anemia | `p(No Anemia) ≥ 0.50` |
| 3 | Folate Deficiency | `p(Folate) ≥ 0.40` |
| 4 | Grey Zone Triage | `p(B12 \| IDA or B12)` falls inside the configured band |
| 2 | B12 Deficiency | `p(B12 \| IDA or B12)` above the band |
| 1 | IDA | otherwise |

The grey zone is a deliberate, honest outcome: when the model's own B12-vs-IDA conditional
sits near the boundary, it says so and asks for review instead of forcing a coin-flip
diagnosis into a treatment decision.

### Persisted artifacts

Everything needed for inference ships in `backend/path_b_blood_report/mod_b3_grader/models/`:

```
xgb_cascade.joblib        three fitted stage estimators + the isotonic calibrator
hetero_gnn_weights.pt     GNN state dict
gnn_metadata.pt           feature/class/hidden dims + the knowledge edge list
standard_scaler.joblib    feature scaling
knn_imputer.joblib        missing-value imputation
pipeline_config.joblib    α and the grey-zone band bounds
```

The model implementation is a faithful reconstruction of the training script in
[docs/model.py](docs/model.py), and the seven-report corpus in
`backend/shared/sample_reports/test_pdfs/` pins that reconstruction — every expected verdict
is recorded in its `MANIFEST.json` and asserted by tests.

---

## Explainability (XAI)

A verdict a clinician cannot interrogate is not usable. VitaScan has two complementary
explanation levels, added in a dedicated sidecar package at
`backend/path_b_blood_report/mod_b3_grader/explainability/`.

The package **reads** the deployed model and never modifies it. Predictions are required to
stay byte-identical to what the model produced before the explainability layer existed, and
the manifest-pinned corpus enforces that.

### Level 1 — which biomarkers drove this verdict?

Attribution runs on both branches and is combined:

| Branch | Method | Units produced |
| --- | --- | --- |
| XGBoost cascade | Exact **TreeSHAP** per stage, composed by chain rule | log-probability |
| Hetero-GNN | **Integrated Gradients** through the patient node | logit |

Because those two are dimensionally incommensurable, the blend is performed on
**normalised shares** — each feature's fraction of the total absolute contribution within
its own branch — not by applying `α` to raw attributions. That makes the shares comparable,
at the cost of exactness, and every result carries a `caveats` list saying so.

Each contribution is annotated with:

- **`estimated`** — the underlying value was KNN-imputed, not measured
- **`is_derived`** / **`derived_from`** — the feature is a computed index, and which measured
  values it was computed from

Attribution is deliberately **not** load-bearing. If SHAP is unavailable, the tree is
unexplainable, or torch disagrees, the explanation is omitted and the verdict still stands —
but the failure is logged loudly, because a silent fallback in a medical tool is how an
invented number ends up back on a patient-facing screen.

Where attribution is unavailable, the results page falls back to ranking the largest measured
deviations, and **labels that fallback distinctly** (`source: "deviation"` vs
`source: "model"`). A clinical heuristic is never presented as a model weight.

### Level 2 — what would move this verdict?

For each **measured** biomarker, a counterfactual search finds the nearest value at which the
decision flips:

- Grid-then-bisect over the marker's plausible range (21 grid samples, bisection to 1e-3 of
  the searched span), because the triage rules are a chain of thresholds and the decision is
  not guaranteed monotonic in any single marker
- Every reported flip is **re-verified** by re-running the model on the proposed row — a
  search can converge on an artefact, and verification is what makes the claim safe
- **Imputed features are refused**, not searched: asking what ferritin value would flip a
  verdict on a report that never measured ferritin invites counselling on a number nobody has
- Biomarkers whose entire plausible range leaves the verdict unchanged are reported as
  `unreachable`, which is itself a real answer

Counterfactuals are **univariate** — one marker moves, everything else is held fixed. Real
biomarkers move together, so each result describes a point the model can be queried at, not
a patient that exists. The plausible ranges are hand-set from clinical reference limits, not
learned from training data, so a flip near a range edge is marked `at_range_boundary` as a
statement about extrapolation. All of this ships in the `caveats` list so it cannot be
silently dropped on the way to the screen.

**Counterfactuals are opt-in**, because the GNN shares biomarker nodes across a batch and
therefore cannot be batched for probing — every probe is a single-row forward pass, costing
seconds per patient rather than milliseconds:

```bash
VITASCAN_COUNTERFACTUALS=1    # enable; absent ≠ empty in the payload
```

### Model verdict card

The results page carries an explicit *Anemia Grader Verdict* card showing the decision, the
blended probability, the XGB and GNN components separately, the blend weight `α`, the
per-feature share breakdown, and the imputation footer when values were estimated.

---

## Path A — symptom photo cross-check

A photograph of a clinical site — eyes, nails, skin, tongue, or hair — is sent to a Groq
vision model, which looks for the visual signs of these deficiencies: conjunctival pallor,
koilonychia (spoon nails), glossitis, skin lesions.

The result is formatted as an independent cross-check signal against the blood-derived
verdict, in either direction: it can corroborate a deficiency or contradict it. It is
explicitly a **corroborating signal**, never a replacement for the blood panel, and the UI
presents it that way.

---

## Diet recommendations (FSSAI / IFCT2017 RAG)

Behind `backend/diet_rag_service/` is a curated, offline knowledge graph — not a vector store
guessing at nutrition facts.

**Knowledge base** (`kb/`):

| File | Contents |
| --- | --- |
| `foods.json` | 84 curated Indian foods with per-100g nutrient values from **IFCT2017** (National Institute of Nutrition, Hyderabad) |
| `nutrients.json` | 20 nutrients with **ICMR-NIN 2020** recommended dietary allowances for Indian adults |
| `deficiencies.json` | 8 deficiency definitions: target nutrients, nutrients to emphasise, and bioavailability tips |
| `allergies.json` | 9 allergen classes mapped to the foods that contain them |
| `disorders.json` | 9 digestive/metabolic disorder rules (lactose intolerance, celiac, IBS, …) |

**Planning pipeline**, degraded by design so it never depends on a network call:

1. The knowledge graph produces a deterministic 7-day meal plan (Breakfast/Lunch/Snack/Dinner
   with per-slot food-group priority) plus the targeted-nutrient explanation
2. Live IFCT2017 data is attached when curated coverage is thin
3. A Groq LLM (JSON mode, model auto-probed from a fallback chain) enriches the plan with
   culturally appropriate dishes — but the deterministic plan remains the guaranteed output

Allergy and disorder filters apply before selection: allergens are excluded, `avoid_tags`
filtered out, `prefer_tags` boosted, and per-nutrient limits honoured. Every suggestion is
FSSAI-checked and carries a warning that automated guidance is not a substitute for a
physician or dietitian.

---

## Reminders

`backend/reminders/` handles supplement and appointment reminders:

- Confirmation email on creation, plus lead-time reminder dispatch
- Brevo API with an SMTP fallback
- Google Calendar deep link and a downloadable `.ics` file generated per reminder
- Supabase-backed storage with an in-memory fallback store, so the feature works without
  Supabase credentials configured
- Due reminders are processed lazily on reminder endpoints rather than by a background worker

---

## Tech stack

**Backend**

| Area | Technology |
| --- | --- |
| API | FastAPI, Uvicorn, Pydantic v2 |
| ML | PyTorch, PyTorch Geometric, XGBoost, scikit-learn (pinned `==1.6.1`), SHAP, NumPy, pandas, joblib |
| NLP / OCR | GLiNER, HuggingFace Transformers (BioBERT), PyMuPDF, pytesseract, Pillow |
| LLM providers | Groq (vision analysis, diet RAG) |
| Persistence | Supabase (`supabase-py`), local JSON artifacts |
| Email | Brevo API, SMTP fallback, `icalendar` |

**Frontend**

| Area | Technology |
| --- | --- |
| Framework | Next.js 14 (App Router), React 18, TypeScript 5.5 |
| Styling | Tailwind CSS 3.4 |
| Auth | Clerk (`@clerk/nextjs`) |
| Charts | Recharts |
| Icons | lucide-react |
| HTTP | Axios |
| i18n | Custom context + dictionary (English, Hindi, Marathi) |

---

## Repository layout

```
Vitascan_V3_Gq/
├── backend/
│   ├── run_pipeline.py                 FastAPI orchestrator (port 8000)
│   ├── path_b_blood_report/            ← the core
│   │   ├── mod_b1_extractor/           PDF/OCR text + biomedical NER
│   │   ├── mod_b2_normalizer/          units, reference ranges, grader payload
│   │   └── mod_b3_grader/              hybrid grader
│   │       ├── grader_model/           cascade, GNN, features, inference
│   │       ├── explainability/         ← XAI sidecar (SHAP + IG + counterfactuals)
│   │       ├── models/                 committed model artifacts
│   │       └── gat_model.py            legacy GAT deficiency grader
│   ├── path_a_symptom_image/           Groq vision cross-check
│   ├── mod_c_explainer/                explanation assembly, FSSAI checks
│   ├── diet_rag_service/               knowledge graph + RAG planner
│   ├── reminders/                      email, calendar, .ics, CRUD
│   └── shared/                         schemas, storage, Supabase client,
│                                       sample_reports/ (7 synthetic PDFs)
├── frontend/                           Next.js 14 App Router
│   ├── app/                            upload, results, diet-planner, reminders, auth
│   ├── components/                     Navbar, ModelVerdictCard, DietRAGChat, …
│   └── lib/                            api client, Supabase, i18n
├── docs/                               architecture, model training, SQL schema
├── Design/                             UI design references
├── RUN_GUIDE.md                        how to run both halves
└── start.py / start.bat / start.sh     one-command launcher
```

---

## Getting started

### Prerequisites

| Requirement | Notes |
| --- | --- |
| **Python 3.12** (x64) | Newer CPython releases have no Windows wheels yet for `torch`, `torch-geometric`, and `gliner`, so pip falls back to building from source |
| **Node.js 18.17+** and npm | |
| **Tesseract OCR 5.x** | *Optional* — only needed for scanned/image-only PDFs. Text-layer reports never call it |

### 1. Backend

```bash
python -m venv backend/venv
source backend/venv/bin/activate          # Windows: .\backend\venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

Copy `backend/.env.example` to `backend/.env` and fill in the Groq keys. **Only the Groq keys
matter for the AI features** — extraction, normalisation, grading, explainability, and the
results payload all run locally.

```env
GROQ_API_KEY=gsk_...
GROQ_VISION_API_KEY=gsk_...
GROQ_RAG_API_KEY=gsk_...
```

Start the server — either directory works, the app resolves `backend/.env` and `sys.path`
itself:

```bash
python -m uvicorn backend.run_pipeline:app --port 8000   # from the repo root
python run_pipeline.py                                   # or from backend/
```

> **First start takes 40–60 seconds.** GLiNER and the torch/PyG GNN are imported and the
> committed model bundle is unpickled on first use.

API on **http://localhost:8000**, interactive docs at **/docs**.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev            # http://localhost:3000
```

`frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

> `NEXT_PUBLIC_*` values are inlined **at build time** — restart the dev server (or rebuild
> before `npm start`) after changing them. Never run `npm run dev` against the same `.next`
> folder as a live `npm start`; the dev server rewrites the production build and the running
> server then serves dead chunks.

### 3. Optional: database

Run [docs/supabase_schema.sql](docs/supabase_schema.sql) in the Supabase SQL editor to
create the `patients`, `scans`, `deficiencies`, and `reminders` tables. Without Supabase
configured, the app degrades gracefully — reminders fall back to an in-memory store and
results are held in process state.

### 4. One-command launch

`start.py` (and the `start.bat` / `start.sh` wrappers) brings up both halves together.

---

## Trying it end to end

Seven synthetic, text-layer blood-report PDFs ship in
`backend/shared/sample_reports/test_pdfs/`. They are real PDFs with real text layers, so
they exercise the entire Path B chain. **All values are synthetic — no real patient data.**

| PDF | Case | Expected verdict |
| --- | --- | --- |
| `IND-01-HEALTHY-MALE.pdf` | normal male, all 9 features present | No Anemia |
| `IND-02-HEALTHY-FEMALE.pdf` | normal female | No Anemia |
| `IND-03-IDA-FEMALE.pdf` | microcytic, ferritin 7.2 ng/mL | IDA |
| `IND-04-MEGALOBLASTIC-B12-MALE.pdf` | macrocytic (MCV 105), B12 84 pg/mL | Grey Zone Triage |
| `IND-05-MEGALOBLASTIC-FOLATE-FEMALE.pdf` | macrocytic, folate 2.7 ng/mL | Folate Deficiency |
| `IND-06-MIXED-GREYZONE-MALE.pdf` | borderline throughout (grey-zone probe) | Grey Zone Triage |
| `IND-07-CBC-ONLY-FEMALE.pdf` | CBC only, **no** ferritin/B12/folate | IDA, 3 features KNN-imputed |

`IND-04` and `IND-06` land in *Grey Zone Triage* by design — that is the model's own triage
threshold, not a misclassification. `IND-07` exists to exercise the imputation path and
triggers the "3 biomarker(s) KNN-imputed" footer on the results page.

Upload one at <http://localhost:3000/upload>, or:

```bash
curl -X POST http://localhost:8000/upload-report \
  -F "file=@backend/shared/sample_reports/test_pdfs/IND-03-IDA-FEMALE.pdf" \
  -F "patient_id=DEMO-01"
```

`MANIFEST.json` records every expected value and verdict, so a run can be diffed
automatically. Note the reports are laid out **one analyte per line** deliberately — the
extractors are line-oriented, and this is documented in that folder's README so it does not
get "fixed".

---

## API reference

**Reports and results**

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Health/handshake |
| `GET` | `/status` | Reports which extractor mode and whether the vision client is live |
| `POST` | `/upload-report` | Full Path B chain: extract → normalise → grade → explain → format |
| `POST` | `/upload-reports-batch` | Multiple report files for one patient, composed into a single model input |
| `POST` | `/upload-symptom-photo` | Path A vision analysis + cross-check signal |
| `GET` | `/results` | Latest formatted results payload |

**Diet**

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/diet/generate-plan` | 7-day plan for a deficiency, diet preference, allergies, disorders |
| `POST` | `/api/diet/chat` | Conversational query against the same knowledge base |

**Reminders**

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/reminders` | Create (sends confirmation email, fires due reminders) |
| `GET` | `/reminders?patient_id=` | List for a patient |
| `DELETE` | `/reminders/{reminder_id}` | Delete |
| `POST` | `/api/reminders/schedule` | Alias of create |

Each upload persists its stage artifacts as JSON:

```
extractions/pdf_raw/         raw extracted text + structure
extractions/pdf_raw_parts/   per-file parts of a batch upload
extractions/grader_input/    the 9-key model payload
extractions/biomarkers/      NER output
extractions/model_grade/     full grader verdict, probabilities, explanation
extractions/groq_symptom/    vision analysis
extractions/diet_plans/      generated plans
extractions/results/         the final front-end payload
```

---

## Testing

**88 tests across 14 files**, run with:

```bash
python -m pytest backend -q
```

| Area | Files | Coverage |
| --- | --- | --- |
| Explainability | `explainability/test_explainability.py`, `test_counterfactual.py`, `test_integration.py` | 36 — attribution composition and residual, counterfactual verification, manifest-pinned verdicts, and the guarantee that explanations never alter the verdict |
| Grader / model | `test_model_grader_india.py` | 1 — end-to-end grading of the 6 composite India scenarios through `GraderInputBuilder` → `PathBGrader` |
| Normaliser | `mod_b2_normalizer/test_grader_input.py`, `test_composer.py` | 16 — payload construction, unit conversion, plausibility gating, composition rules |
| Extraction | `shared/test_extraction_biobert.py` | 5 |
| Schemas | `shared/test_schemas.py`, `reminders/test_schemas.py` | 5 |
| Diet RAG | `diet_rag_service/test_diet_rag.py` | 9 |
| Path A | `path_a_symptom_image/test_crosscheck.py` | 3 |
| Reminders | `test_calendar_link.py`, `test_emailer.py`, `test_reminders_service.py` | 13 |

Frontend type-check:

```bash
cd frontend && npx tsc --noEmit
```

---

## Design decisions and known limits

This section exists because a system like this is only trustworthy if it is explicit about
what it does not claim.

**Decision-making**

- **Grey zone is a first-class outcome.** When the B12-vs-IDA conditional sits near the
  boundary, the model reports triage rather than guessing. A forced answer would be worse
  than an honest referral.
- **The grey zone costs recall on clean cases.** `IND-04`, a textbook macrocytic B12 case,
  lands in triage. That is the triage rule working as configured, not a bug, but it is a real
  trade-off in how the thresholds are set.
- **KNN imputation always permits a verdict.** The model grades reports that omit ferritin,
  B12, or folate. Every imputed value is disclosed on the results page and marked per-feature
  in the explanation.

**Explanation integrity**

- **Shares are not probabilities.** A contribution share says how much of the *explanation* a
  biomarker carries — never how likely the verdict is. The schemas and the UI both state this.
- **The share blend is not the model's α blend.** The two branches produce attributions in
  different units, so they are blended as normalised shares. This is a deliberate
  approximation and every result carries the caveat.
- **Counterfactuals describe the boundary, not a treatment path.** They are univariate and
  the ranges are hand-set rather than learned, so they show where the decision boundary lies
  — not what would happen to a real patient.
- **Imputed markers are refused in counterfactual search**, because the report cannot
  support the question.
- **Explanations never gate a verdict.** Attribution and counterfactual failures degrade to
  "no explanation" and are logged loudly.

**Numbers that were removed rather than faked**

Two fields previously carried hardcoded literals — a "GAT attention weights" triple and a
"baseline model benchmark" comparing logistic regression, random forest, and XGBoost — both
identical for every patient and rendered as though derived from the report. Neither was ever
computed. They were **deleted**, not restyled:

- `ModB3Output.attention_weights` is deprecated and always empty; real attribution lives on
  `model.explanation`
- `ModB3Output.baseline_comparison` is always empty, and its schema description instructs
  that it is not to be populated except from a real held-out evaluation

**Other known limits**

- `GET /results` serves **clearly-synthetic demo data** when no report has been uploaded in
  the current session, so the dashboard is explorable before any upload. It contains invented
  lab values and is not a real assessment.
- The frontend's optional Supabase persistence helpers are not exercised by the automated
  tests.
- Baseline model comparison (logistic regression / random forest / XGBoost benchmarks) has
  **not been evaluated**. That is real research work against a labelled hold-out split, and
  it knowingly remains undone rather than approximated.

---

## Documentation

| Document | Contents |
| --- | --- |
| [RUN_GUIDE.md](RUN_GUIDE.md) | Step-by-step run instructions for both halves |
| [WINDOWS_SETUP.md](WINDOWS_SETUP.md) | Windows specifics: Python version, PowerShell activation, Tesseract, the dev/start gotcha |
| [docs/ARCHITECTURE_AND_SETUP.md](docs/ARCHITECTURE_AND_SETUP.md) | Module-by-module architecture |
| [docs/model.py](docs/model.py) | The training script the deployed model was reconstructed from |
| [docs/supabase_schema.sql](docs/supabase_schema.sql) | Database schema |
| [docs/VISION_PATH_TEST_REPORT.md](docs/VISION_PATH_TEST_REPORT.md) | Path A evaluation |
| [docs/DIET_RAG_TEST_REPORT.md](docs/DIET_RAG_TEST_REPORT.md) | Diet RAG evaluation |
| [backend/shared/sample_reports/test_pdfs/README.md](backend/shared/sample_reports/test_pdfs/README.md) | The test corpus and its layout constraints |

---

