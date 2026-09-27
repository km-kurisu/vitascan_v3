# VitaScan V3 — How to Run Backend & Frontend

This guide provides step-by-step instructions to run the backend FastAPI service
and frontend Next.js web application.

> **Windows?** See [WINDOWS_SETUP.md](WINDOWS_SETUP.md) — Python version, PowerShell
> activation, Tesseract, and the `npm run dev` / `npm start` gotcha.

---

## 🚀 Quick Start Summary

- **Backend**: `cd backend` → `python run_pipeline.py` (http://localhost:8000)
- **Frontend**: `cd frontend` → `npm run dev` (http://localhost:3000)

---

## 1. Prerequisites

1. **Python 3.12** (x64). Newer CPython releases have no Windows wheels yet for
   the ML stack (`torch`, `torch-geometric`, `gliner`), so `pip` falls back to
   building from source. Check with `python --version`.
2. **Node.js 18.17+ & npm**. Check with `node -v`.
3. **Tesseract OCR 5.x** — *optional*, only needed for scanned/image-only PDFs.
   The blood-report pipeline reads text-layer PDFs (like everything in
   `backend/shared/sample_reports/test_pdfs/`) with PyMuPDF and never calls it.

---

## 2. Environment Configuration

### Backend (`backend/.env`)

Copy `backend/.env.example` to `backend/.env`. Only the Groq keys matter for the
AI features — extraction, normalization, grading and the results payload run
locally:

```env
HOST=0.0.0.0
PORT=8000
DEBUG=True

GROQ_API_KEY=gsk_...
GROQ_VISION_API_KEY=gsk_...
GROQ_RAG_API_KEY=gsk_...

# optional
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-key
BREVO_API_KEY=xkeysib-...
BREVO_FROM=noreply@vitascan.ai
```

### Frontend (`frontend/.env.local`)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

`NEXT_PUBLIC_*` values are inlined **at build time**: restart `npm run dev`, or
rebuild before `npm start`, after changing them. A Clerk publishable key is not
required (`components/Providers.tsx` falls back to a placeholder).

---

## 3. Running the Backend

From the repo root:

```bash
python -m venv backend/venv
source backend/venv/bin/activate        # Windows: .\backend\venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

Then start the server (either directory works — the app fixes `sys.path` and
resolves `backend/.env` itself):

```bash
# a) from the repo root
python -m uvicorn backend.run_pipeline:app --port 8000

# b) from backend/  (equivalent)
cd backend
python run_pipeline.py
```

FastAPI starts on **http://localhost:8000**; interactive docs at
**http://localhost:8000/docs**. The first start takes ~40–60 s: the GLiNER NER
model and the torch/PyG heterogeneous GNN are imported, and the committed
`models/` bundle (`xgboost` + `scikit-learn` pickles) is unpickled on first use.

## 4. Running the Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

> Never run `npm run dev` against the same `.next` folder as a live
> `npm start` — the dev server rewrites the production build and the running
> server then serves dead chunks (unstyled page).

---

## 5. System Features & API Flow

1. **Blood Report PDF Extraction (`/upload-report`)**:
   - PyMuPDF text-stream extraction + Tesseract OCR preprocessing for scanned pages.
   - GLiNER biomedical NER model (`Ihor/gliner-biomed-small-v1.0`) + regex fallback
     for biomarker value detection, then deviation normalization.
   - `mod_b2_normalizer` assembles the 9-key grader payload (missing biomarkers stay
     `None` and are KNN-imputed by the model).
   - `mod_b3_grader` runs the hybrid grader (XGBoost cascade + heterogeneous GNN,
     alpha-blended) and logs a `model_grade/` artifact.
   - `mod_c_explainer` renders the Mod C payload consumed by the results page,
     including the *Anemia Grader Verdict* card data.
2. **Physical Symptom Analysis (`/upload-symptom-photo`)**:
   - Groq Multimodal Vision API (`llama-3.2-11b-vision-preview`) via `GROQ_VISION_API_KEY`.
   - Cross-references clinical visual signs against blood biomarker deficiencies.
3. **FSSAI Diet RAG Engine (`/api/diet/generate-plan`, `/api/diet/chat`)**:
   - Groq LLaMA 3.3 70B RAG engine (`GROQ_RAG_API_KEY`) over the FSSAI RDA knowledge base.
4. **Supplement Reminders (`/reminders`)**:
   - Brevo HTML emails with Google Calendar and `.ics` download link generators.

---

## 6. Trying it end-to-end

Upload one of the committed synthetic reports from
`backend/shared/sample_reports/test_pdfs/` at <http://localhost:3000/upload>, or:

```bash
curl -X POST http://localhost:8000/upload-report \
  -F "file=@backend/shared/sample_reports/test_pdfs/IND-03-IDA-FEMALE.pdf" \
  -F "patient_id=DEMO-01"
```

Expected verdicts for all seven reports are in
`backend/shared/sample_reports/test_pdfs/MANIFEST.json`
(`IND-03` → IDA, `IND-07` → IDA with 3 KNN-imputed features, …).

Run the tests:

```bash
python -m pytest backend -q
```
