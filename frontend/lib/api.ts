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
  /** Share of the explanation this biomarker carries, as a percentage. */
  impact_pct: number;
  direction?: string;
  /** True when the underlying value was imputed, not measured in the report. */
  estimated?: boolean;
  /**
   * 'model' when derived from real model attribution, 'deviation' when falling
   * back to the largest measured deviations. These are not equivalent and must
   * not be rendered identically: only 'model' reflects what the model used.
   */
  source?: 'model' | 'deviation' | string;
}

export interface FeatureAttribution {
  feature: string;
  label: string;
  /**
   * Signed share of the explanation, as a percentage. Shares sum to 100 in
   * absolute value. This is a share, not a probability -- it never says how
   * likely the verdict is.
   */
  share_pct: number;
  /** True when the underlying value was imputed, not measured. */
  estimated?: boolean;
  is_derived?: boolean;
  derived_from?: string[];
}

export interface ModelExplanation {
  explained_class: number;
  explained_label: string;
  contributions: FeatureAttribution[];
  /** p(B12 | IDA or B12); the quantity the grey-zone band tests. */
  p_b12_conditional?: number | null;
  grey_zone_band?: number[] | null;
  /** False when only the XGB cascade could be explained. */
  gnn_available?: boolean;
  /** Limits of this explanation, carried through so the UI can show them. */
  caveats?: string[];
}

export interface CounterfactualTarget {
  feature: string;
  label: string;
  /** Reporting unit, e.g. ng/mL. Present so a bare number is never ambiguous. */
  unit?: string;
  current_value: number;
  target_value: number;
  direction: 'increase' | 'decrease' | string;
  /** The verdict on the other side of this boundary. */
  to_etiology: string;
  /** True when the boundary sits at the edge of the searched range. */
  at_range_boundary?: boolean;
}

export interface CounterfactualExplanation {
  /** Nearest boundary first. Empty when nothing moves this verdict. */
  targets: CounterfactualTarget[];
  /** Biomarkers whose whole plausible range leaves the verdict unchanged. */
  unreachable?: string[];
  /** Biomarkers not searched because the value was estimated, not measured. */
  refused?: string[];
  caveats?: string[];
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
  /**
   * Real per-feature attribution for this verdict. Absent when explanation is
   * disabled or the attribution stack is unavailable -- render a fallback
   * rather than assuming it is present.
   */
  explanation?: ModelExplanation | null;
  /**
   * What would move this verdict, by how much. Absent unless the server was
   * asked to compute it -- the search costs seconds. Absent is not the same as
   * empty: empty means the search ran and found nothing that moves the verdict.
   */
  counterfactuals?: CounterfactualExplanation | null;
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
