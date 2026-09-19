# VitaScan V3 (Groq API Edition) — Technical Architecture & Setup Guide

## 1. Overview
VitaScan V3 is an AI-powered multi-modal biomarker deficiency detection platform that integrates:
1. **Blood Report Processing (Path B)**: PyMuPDF + Tesseract OCR + GLiNER Biomedical NER (Ihor/gliner-biomed-small-v1.0) + Reference Range Normalizer + GAT Model (gat_model.py).
2. **Physical Symptom Vision Analysis (Path A)**: Groq Multimodal Vision API (llama-3.2-11b-vision-preview) with GROQ_VISION_API_KEY.
3. **FSSAI Diet RAG Engine**: Groq LLaMA 3.3 70B RAG (GROQ_RAG_API_KEY) querying FSSAI RDA Knowledge Base (ssai_knowledge_base.json).
4. **Email & Google Calendar Reminders**: Brevo SMTP/API HTML email dispatch + Google Calendar link & .ics file generator.
5. **Next.js 14 Web Frontend**: Interactive dark-themed UI matching Design/ specs with i18n support, deficiency detail pages, diet planner, and reminder modal.

---

## 2. Directory Structure

`
Vitascan_V3_Gq/
├── backend/
│   ├── run_pipeline.py                 # FastAPI orchestrator on port 8000
│   ├── .env                            # Dual Groq keys, Tesseract path, Brevo, Supabase
│   ├── path_b_blood_report/
│   │   ├── mod_b1_extractor/           # PyMuPDF + Tesseract OCR + GLiNER extraction
│   │   ├── mod_b2_normalizer/          # Reference ranges & non-linear normalizer
│   │   └── mod_b3_grader/              # VitaScanGAT architecture & .pkl loader
│   ├── path_a_symptom_image/           # Groq LLaMA 3.2 11B Vision analyzer
│   ├── diet_rag_service/               # FSSAI RDA knowledge base & RAG engine
│   ├── reminders/                      # Emailer, calendar link builder, CRUD router
│   ├── mod_c_explainer/                # Groq clinical explainer & formatter
│   └── shared/                         # Schemas, storage, Supabase client
├── frontend/                           # Next.js 14 App Router application
│   ├── app/                            # Upload, Results, Diet Planner, Reminders
│   ├── components/                     # Navbar, Footer, LanguageDropdown, ReminderModal
│   └── lib/i18n/                       # Translation dictionaries (EN, ES, HI, TA, TE)
├── docs/
│   ├── supabase_schema.sql             # SQL script for patients, scans, deficiencies, reminders
│   ├── ARCHITECTURE_AND_SETUP.md       # Complete architecture documentation
│   └── RUN_GUIDE.md                    # Quick startup guide
└── RUN_GUIDE.md                        # Root execution guide
`

---

## 3. How to Run

### Backend (FastAPI)
`ash
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
`
*Server runs on http://localhost:8000 (Docs at http://localhost:8000/docs)*

### Frontend (Next.js)
`ash
cd frontend
npm install
npm run dev
`
*App runs on http://localhost:3000*

---

## 4. Supabase Database Setup
Execute [docs/supabase_schema.sql](file:///i:/Code/Vitascan_V3_Gq/docs/supabase_schema.sql) in your Supabase SQL Editor to initialize:
- patients
- scans
- deficiencies
- 
eminders
