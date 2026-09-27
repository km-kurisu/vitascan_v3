# Running VitaScan V3 on Windows

Verified setup notes: PowerShell, Python **3.12** (x64), Node **20 LTS**.
The backend was developed on Linux/macOS, so this file lists the places where
Windows needs something different.

```powershell
git clone https://github.com/km-kurisu/vitascan_v3.git
cd vitascan_v3
```

---

## 1. Prerequisites

| Tool | Version | Notes |
| --- | --- | --- |
| Python | **3.12 x64** | see the warning below — do not use 3.13/3.14 |
| Node.js | 20 LTS (>= 18.17) | `node -v` |
| Tesseract OCR | 5.x | **optional**, only for scanned/image PDFs |

### Python version matters

The ML stack (`torch`, `torch-geometric`, `gliner`, `transformers`) needs binary
wheels. Wheels for the newest CPython releases lag badly on Windows, so
`pip install torch` can spend 20+ minutes building from source — or fail outright.
**Python 3.12 installs from wheels in a couple of minutes.** The committed
`backend/venv` here happens to be 3.14 and works on Linux, but that is not the
version to install on Windows.

### Tesseract is optional

`BloodReportExtractor` reads text-layer PDFs with PyMuPDF, and only shells out to
Tesseract for scanned pages. Every report in
`backend/shared/sample_reports/test_pdfs/` has a text layer, so the whole blood
pipeline works without Tesseract. If you need OCR for scanned reports, install
[UB Mannheim's build](https://github.com/UB-Mannheim/tesseract/wiki) and set
`TESSERACT_CMD` in `backend/.env` (the code also probes
`C:\Program Files\Tesseract-OCR\tesseract.exe` and the `%LOCALAPPDATA%` paths).

---

## 2. Backend

```powershell
# from the repo root
py -3.12 -m venv backend\venv
.\backend\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r backend\requirements.txt
```

If PowerShell refuses to activate the venv (`running scripts is disabled`):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -RemoteSigned
```

### Start it

Either of these works — the app fixes `sys.path` and the `.env` path itself, so
the working directory does not matter:

```powershell
# a) from the repo root
python -m uvicorn backend.run_pipeline:app --port 8000

# b) from backend/  (equivalent)
cd backend
python run_pipeline.py
```

Check it: <http://localhost:8000/status> and <http://localhost:8000/docs>.
First start is slow (~40-60 s): the GLiNER NER model and the torch/PyG
heterogeneous GNN are imported and the `models/` bundle is unpickled on first use.

### `backend/.env`

`backend/.env` is gitignored, so create it from `backend/.env.example`. Only the
Groq keys are needed for the AI features; the blood-report pipeline (extraction →
normalization → grader → results) runs entirely locally:

```dotenv
GROQ_API_KEY=gsk_...
GROQ_VISION_API_KEY=gsk_...
GROQ_RAG_API_KEY=gsk_...

# optional
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=...
BREVO_API_KEY=xkeysib-...
```

Without Groq keys the server still starts; only the symptom-photo analysis, diet
RAG and Mod C narrative sections degrade.

### Model bundle

`backend/path_b_blood_report/mod_b3_grader/models/` is committed (joblib + `.pt`
files), so the hybrid grader needs no training step — but it *does* need
**xgboost** and **scikit-learn** at unpickle time. Both are in
`backend/requirements.txt`. If a pickle refuses to load, you almost certainly
have a scikit-learn version mismatch with the bundle; reinstall the pinned
version printed in the warning:

```
... InconsistentVersionWarning: Trying to unpickle estimator CalibratedClassifierCV
    from version 1.6.1 when using version X ...
```

---

## 3. Frontend

```powershell
cd frontend
npm install
```

Copy `frontend/.env.example` to `frontend/.env.local` and set:

```dotenv
NEXT_PUBLIC_API_URL=http://localhost:8000
```

`NEXT_PUBLIC_*` values are inlined **at build time**, so if you change the API
URL you must restart (dev) or rebuild (`npm run build`) — a running `npm start`
will keep using the old value.

```powershell
npm run dev      # http://localhost:3000
```

A Clerk publishable key is **not** required: `components/Providers.tsx` falls back
to a placeholder, and `/upload` and `/results` work signed-out.

> Do not run `npm run dev` and `npm start` against the same `.next` folder —
> the dev server rewrites the production build and the running `next start`
> serves dead CSS/JS chunks (unstyled page).

---

## 4. Try it end-to-end

With both servers up, upload one of the committed synthetic reports
(`backend/shared/sample_reports/test_pdfs/`) at <http://localhost:3000/upload>,
or straight from the shell:

```powershell
curl.exe -X POST http://localhost:8000/upload-report `
  -F "file=@backend\shared\sample_reports\test_pdfs\IND-03-IDA-FEMALE.pdf" `
  -F "patient_id=WIN-TEST-01"
```

The response carries `model` (blended XGB cascade + hetero-GNN probabilities),
`model_confidence` and the severity list; `/results` then renders the
*Anemia Grader Verdict* card. `IND-03` should come back as **IDA** — the
expectations for all seven reports are in
`backend/shared/sample_reports/test_pdfs/MANIFEST.json`.

Regenerate the PDFs (needs Chrome or Edge on PATH):

```powershell
.\backend\venv\Scripts\python.exe backend\shared\sample_reports\build_test_pdfs.py
```

---

## 5. Tests

```powershell
.\backend\venv\Scripts\python.exe -m pytest backend -q
```

The India composite grader test (`mod_b3_grader/test_model_grader_india.py`) is
the useful one: it asserts the six reference patients reproduce their expected
verdict.

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'backend'` | you ran `python run_pipeline.py` from the wrong directory — use the repo root or `cd backend` first |
| `ModuleNotFoundError: xgboost` | `pip install -r backend\requirements.txt` (xgboost unpickles the committed cascade) |
| `pip` spends forever on `torch` | wrong Python version — recreate the venv with `py -3.12` |
| Frontend fetches fail / spinner forever | `NEXT_PUBLIC_API_URL` points elsewhere, or the backend isn't on :8000; rebuild after changing it |
| Page renders unstyled | you ran `next dev` over a live `next start`; `rm -r .next` / rebuild |
| First upload takes ~1 min, then is fast | expected: lazy model load (GLiNER + torch/PyG) |
| `No Anemia` verdict for a report you expect to be anaemic | the extractors are line-oriented; see the layout constraints in `test_pdfs/README.md` |
