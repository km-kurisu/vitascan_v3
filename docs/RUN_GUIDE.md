# VitaScan V3 — How to Run Backend & Frontend

This guide provides step-by-step instructions to run the backend FastAPI service and frontend Next.js web application.

---

## 🚀 Quick Start Summary

- **Backend**: cd backend -> python run_pipeline.py (Runs on http://localhost:8000)
- **Frontend**: cd frontend -> 
pm run dev (Runs on http://localhost:3000)

---

## 1. Prerequisites

1. **Python 3.10+**: Make sure Python is installed (python --version).
2. **Node.js 18+ & npm**: Make sure Node is installed (
ode -v).
3. **Tesseract OCR**: Installed at C:\Program Files\Tesseract-OCR\tesseract.exe (or in system PATH).

---

## 2. Environment Configuration

### Backend (ackend/.env)
Ensure ackend/.env is configured with required keys:
`nv
HOST=0.0.0.0
PORT=8000
DEBUG=True

GROQ_API_KEY=gsk_...
GROQ_VISION_API_KEY=gsk_...
GROQ_RAG_API_KEY=gsk_...

TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# Supabase Credentials (Optional)
SUPABASE_URL=https://your-supabase-project.supabase.co
SUPABASE_KEY=your-supabase-key

# Email & Reminders (Brevo API)
BREVO_API_KEY=xkeysib-...
BREVO_FROM=noreply@vitascan.ai
`

### Frontend (rontend/.env.local)
`nv
NEXT_PUBLIC_API_URL=http://localhost:8000
`

---

## 3. Running the Backend

Open terminal in ackend/ and execute:

`ash
# 1. Navigate to backend
cd backend

# 2. Activate virtual environment (optional)
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start backend server
python run_pipeline.py
`

FastAPI server starts on **http://localhost:8000**.  
Interactive API Documentation (Swagger UI) is available at **http://localhost:8000/docs**.

---

## 4. Running the Frontend

Open terminal in rontend/ and execute:

`ash
# 1. Navigate to frontend
cd frontend

# 2. Install dependencies
npm install

# 3. Start development server
npm run dev
`

Next.js application starts on **http://localhost:3000**.

---

## 5. System Features & API Flow

1. **Blood Report PDF Extraction (/upload-report)**:
   - PyMuPDF text stream extraction + Tesseract OCR preprocessing for scanned pages.
   - GLiNER biomedical NER model (Ihor/gliner-biomed-small-v1.0) + Regex fallback for biomarker value detection.
   - Deviation normalization & GAT model evaluation.

2. **Physical Symptom Analysis (/upload-symptom-photo)**:
   - Groq Multimodal Vision API (llama-3.2-11b-vision-preview) utilizing GROQ_VISION_API_KEY.
   - Cross-references clinical visual signs against blood test biomarker deficiencies.

3. **FSSAI Diet RAG Engine (/api/diet/generate-plan & /api/diet/chat)**:
   - Groq LLaMA 3.3 70B RAG engine (GROQ_RAG_API_KEY) querying FSSAI RDA knowledge base.

4. **Supplement Reminders (/reminders)**:
   - Brevo HTML emails with Google Calendar and .ics download link generators.
