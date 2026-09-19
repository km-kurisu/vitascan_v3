import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

export async function fetchResults() {
  const res = await api.get("/results");
  return res.data;
}

export async function scheduleReminder(data: any) {
  const res = await api.post("/api/reminders/schedule", data);
  return res.data;
}

export async function generateDietPlan(patientId: string, deficiencyType: string, dietPref: string) {
  const res = await api.post("/api/diet/generate-plan", {
    patient_id: patientId,
    deficiency_type: deficiencyType,
    diet_preference: dietPref
  });
  return res.data;
}

export async function sendDietChatQuery(query: string, deficiencyType: string) {
  const res = await api.post("/api/diet/chat", { query, deficiency_type: deficiencyType });
  return res.data;
}

export { uploadReport as uploadBloodReport };
