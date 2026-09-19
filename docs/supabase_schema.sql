-- VitaScan Supabase Complete Database Schema
-- Execute this script in your Supabase SQL Editor (https://supabase.com/dashboard/project/_/sql)

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==========================================
-- 1. Patients Table
-- ==========================================
CREATE TABLE IF NOT EXISTS public.patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id VARCHAR(50) UNIQUE NOT NULL,
    clerk_user_id VARCHAR(255),
    full_name VARCHAR(255),
    age INT,
    gender VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================
-- 2. Scans / Report Runs Table
-- ==========================================
CREATE TABLE IF NOT EXISTS public.scans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scan_id VARCHAR(100) UNIQUE NOT NULL,
    patient_id VARCHAR(50) REFERENCES public.patients(patient_id) ON DELETE CASCADE,
    blood_report_url TEXT,
    symptom_photo_url TEXT,
    overall_risk_band VARCHAR(20) DEFAULT 'unknown',
    flagged_count INT DEFAULT 0,
    mod_c_output JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================
-- 3. Deficiencies Breakdown Table
-- ==========================================
CREATE TABLE IF NOT EXISTS public.deficiencies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scan_id VARCHAR(100) REFERENCES public.scans(scan_id) ON DELETE CASCADE,
    deficiency_type VARCHAR(50) NOT NULL, -- 'iron', 'b12', 'anemia', 'folate'
    severity_band VARCHAR(20) NOT NULL,   -- 'severe', 'moderate', 'mild', 'none'
    score_pct NUMERIC(5, 2),
    explanation TEXT,
    key_contributors JSONB,
    diet_recommendations JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================
-- 4. Follow-up Reminders Table
-- ==========================================
CREATE TABLE IF NOT EXISTS public.reminders (
    id VARCHAR(100) PRIMARY KEY,
    patient_id VARCHAR(50) NOT NULL,
    email VARCHAR(255) NOT NULL,
    title VARCHAR(255) NOT NULL,
    notes TEXT,
    appointment_at TIMESTAMP WITH TIME ZONE NOT NULL,
    lead_minutes INT DEFAULT 60,
    status VARCHAR(50) DEFAULT 'confirmed',
    confirmation_sent_at TIMESTAMP WITH TIME ZONE,
    reminder_sent_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================
-- Indexes for Efficient Query Performance
-- ==========================================
CREATE INDEX IF NOT EXISTS idx_patients_clerk_user_id ON public.patients(clerk_user_id);
CREATE INDEX IF NOT EXISTS idx_scans_patient_id ON public.scans(patient_id);
CREATE INDEX IF NOT EXISTS idx_deficiencies_scan_id ON public.deficiencies(scan_id);
CREATE INDEX IF NOT EXISTS idx_reminders_patient_id ON public.reminders(patient_id);

-- ==========================================
-- Enable Row Level Security (RLS)
-- ==========================================
ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.scans ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.deficiencies ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reminders ENABLE ROW LEVEL SECURITY;

-- Clean up existing policies
DROP POLICY IF EXISTS "Allow read patients" ON public.patients;
DROP POLICY IF EXISTS "Allow insert patients" ON public.patients;
DROP POLICY IF EXISTS "Allow update patients" ON public.patients;

DROP POLICY IF EXISTS "Allow read scans" ON public.scans;
DROP POLICY IF EXISTS "Allow insert scans" ON public.scans;
DROP POLICY IF EXISTS "Allow update scans" ON public.scans;

DROP POLICY IF EXISTS "Allow read deficiencies" ON public.deficiencies;
DROP POLICY IF EXISTS "Allow insert deficiencies" ON public.deficiencies;
DROP POLICY IF EXISTS "Allow update deficiencies" ON public.deficiencies;

DROP POLICY IF EXISTS "Allow read reminders" ON public.reminders;
DROP POLICY IF EXISTS "Allow insert reminders" ON public.reminders;
DROP POLICY IF EXISTS "Allow update reminders" ON public.reminders;
DROP POLICY IF EXISTS "Allow delete reminders" ON public.reminders;

-- Define RLS Access Policies
CREATE POLICY "Allow read patients" ON public.patients FOR SELECT USING (true);
CREATE POLICY "Allow insert patients" ON public.patients FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow update patients" ON public.patients FOR UPDATE USING (true);

CREATE POLICY "Allow read scans" ON public.scans FOR SELECT USING (true);
CREATE POLICY "Allow insert scans" ON public.scans FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow update scans" ON public.scans FOR UPDATE USING (true);

CREATE POLICY "Allow read deficiencies" ON public.deficiencies FOR SELECT USING (true);
CREATE POLICY "Allow insert deficiencies" ON public.deficiencies FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow update deficiencies" ON public.deficiencies FOR UPDATE USING (true);

CREATE POLICY "Allow read reminders" ON public.reminders FOR SELECT USING (true);
CREATE POLICY "Allow insert reminders" ON public.reminders FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow update reminders" ON public.reminders FOR UPDATE USING (true);
CREATE POLICY "Allow delete reminders" ON public.reminders FOR DELETE USING (true);
