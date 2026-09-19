import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || "https://placeholder.supabase.co";
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "placeholder";

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

export async function fetchPatientProfileFromSupabase(userId: string) {
  try {
    const { data, error } = await supabase
      .from('patients')
      .select('*')
      .eq('user_id', userId)
      .single();
    if (error) return null;
    return data;
  } catch (err) {
    return null;
  }
}

export async function saveScanToSupabase(scanData: any) {
  try {
    const { data, error } = await supabase
      .from('scans')
      .insert([
        {
          patient_id: scanData.patient_id || 'PAT-DEMO123',
          scan_type: scanData.scan_type || 'blood_report',
          results: scanData.results || scanData,
          created_at: new Date().toISOString(),
        }
      ])
      .select();
    if (error) {
      console.warn('Supabase save error:', error.message);
      return null;
    }
    return data;
  } catch (err) {
    console.warn('Supabase save exception:', err);
    return null;
  }
}

export async function fetchScansFromSupabase(patientId: string) {
  try {
    const { data, error } = await supabase
      .from('scans')
      .select('*')
      .eq('patient_id', patientId)
      .order('created_at', { ascending: false });
    if (error) return [];
    return data;
  } catch (err) {
    return [];
  }
}
