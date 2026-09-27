import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface PatientInfo {
  patient_id: string;
  age?: number;
  gender?: string;
}

export interface SeverityDetail {
  band: string;
  score_pct: string;
  badge_color?: string;
}

export interface KeyContributor {
  biomarker: string;
  impact_pct: number;
  direction?: string;
}

export interface FrontendCrosscheck {
  available: boolean;
  agrees: boolean;
  source: string;
}

export interface DietRecommendation {
  suggestion: string;
  fssai_checked?: boolean;
}

export interface DeficiencyRecommendations {
  foods?: string;
  sunlight?: string;
  tips?: string;
  supplements?: string;
  lifestyle?: string;
  avoid?: string;
}

export interface DeficiencyItem {
  id?: string;
  type: string;
  title?: string;
  severity: SeverityDetail;
  explanation: string;
  key_contributors: KeyContributor[];
  crosscheck: FrontendCrosscheck;
  diet_recommendations: DietRecommendation[];
  recommendations?: DeficiencyRecommendations;
  /** True when the anemia grader verdict implicates this deficiency. */
  model_confirmed?: boolean;
  etiology?: string | null;
  model_confidence?: number | null;
}

export interface SummaryInfo {
  flagged_deficiency_count: number;
  overall_risk_band: string;
}

export interface BloodParameter {
  key: string;
  name: string;
  value: number;
  unit?: string;
  normal_range: string;
  status: 'Low' | 'Normal' | 'High' | 'Borderline' | string;
  deviation?: number;
  /** True when this biomarker feeds the anemia grader. */
  model_input?: boolean;
  confidence?: number;
}

export interface UploadedReport {
  filename?: string;
  uploaded_at?: string;
  page_count?: number | null;
  parse_confidence?: number | null;
}

export interface ModelGradeDetail {
  decision_code: number;
  etiology: string;
  na_count: number;
  complete_input: boolean;
  proba: Record<string, number>;
  proba_xgb: Record<string, number>;
  proba_gnn: Record<string, number>;
  alpha: number;
  imputed_features?: Record<string, number> | null;
}

export interface ModCFrontendOutput {
  generated_at: string;
  schema_version?: string;
  patient: PatientInfo;
  deficiencies: DeficiencyItem[];
  summary: SummaryInfo;
  model?: ModelGradeDetail | null;
  model_confidence?: number | null;
  blood_parameters: BloodParameter[];
  uploaded_report?: UploadedReport | null;
}

export const api = axios.create({
  baseURL: API_BASE_URL,
});

export async function uploadReport(file: File, patientId: string = "PAT-DEMO123") {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("patient_id", patientId);

  const res = await api.post("/upload-report", formData, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return res.data;
}

export async function uploadSymptomPhoto(file: File, sourceRegion: string = "eyes", patientId: string = "PAT-DEMO123") {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("source_region", sourceRegion);
  formData.append("patient_id", patientId);

  const res = await api.post("/upload-symptom-photo", formData, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return res.data;
}

export async function fetchResults(): Promise<ModCFrontendOutput> {
  const res = await api.get("/results");
  return res.data as ModCFrontendOutput;
}

export async function scheduleReminder(data: any) {
  const res = await api.post("/api/reminders/schedule", data);
  return res.data;
}

export async function generateDietPlan(
  patientId: string,
  deficiencyType: string,
  dietPref: string,
  allergies: string[] = [],
  disorders: string[] = []
) {
  const res = await api.post("/api/diet/generate-plan", {
    patient_id: patientId,
    deficiency_type: deficiencyType,
    diet_preference: dietPref,
    allergies,
    disorders
  });
  return res.data;
}

export async function sendDietChatQuery(
  query: string,
  deficiencyType: string,
  dietPref: string = "vegetarian",
  allergies: string[] = [],
  disorders: string[] = []
) {
  const res = await api.post("/api/diet/chat", {
    query,
    deficiency_type: deficiencyType,
    diet_preference: dietPref,
    allergies,
    disorders
  });
  return res.data;
}

export { uploadReport as uploadBloodReport };
