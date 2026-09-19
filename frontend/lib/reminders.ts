import { api } from './api';

export interface Reminder {
  id: string;
  patient_id: string;
  email: string;
  title: string;
  notes?: string;
  appointment_at: string;
  lead_minutes: number;
  status: string;
  calendar_link?: string;
  created_at?: string;
}

export const LEAD_OPTIONS = [
  { label: '1 hour before', minutes: 60 },
  { label: '2 hours before', minutes: 120 },
  { label: '24 hours (1 day) before', minutes: 1440 },
  { label: '48 hours (2 days) before', minutes: 2880 },
  { label: '1 week before', minutes: 10080 },
];

export async function fetchReminders(patientId: string): Promise<Reminder[]> {
  try {
    const res = await api.get('/reminders', { params: { patient_id: patientId } });
    if (res.data && Array.isArray(res.data.reminders)) {
      return res.data.reminders.map((r: any) => ({
        id: r.id || r.reminder_id,
        patient_id: r.patient_id,
        email: r.email,
        title: r.title || r.supplement_name || 'Supplement Routine',
        notes: r.notes || r.dosage || '',
        appointment_at: r.appointment_at || r.scheduled_at || new Date().toISOString(),
        lead_minutes: r.lead_minutes || 1440,
        status: r.status || 'confirmed',
        calendar_link: r.calendar_link || r.google_calendar_url,
      }));
    }
    if (Array.isArray(res.data)) {
      return res.data.map((r: any) => ({
        id: r.id || r.reminder_id,
        patient_id: r.patient_id,
        email: r.email,
        title: r.title || r.supplement_name || 'Supplement Routine',
        notes: r.notes || r.dosage || '',
        appointment_at: r.appointment_at || r.scheduled_at || new Date().toISOString(),
        lead_minutes: r.lead_minutes || 1440,
        status: r.status || 'confirmed',
        calendar_link: r.calendar_link || r.google_calendar_url,
      }));
    }
    return [];
  } catch (err) {
    console.warn('Error fetching reminders from API:', err);
    return [];
  }
}

export async function createReminder(data: {
  patient_id: string;
  email: string;
  title: string;
  notes?: string;
  appointment_at: string;
  lead_minutes: number;
}): Promise<Reminder> {
  const payload = {
    patient_id: data.patient_id,
    email: data.email,
    title: data.title,
    notes: data.notes || '',
    appointment_at: data.appointment_at,
    lead_minutes: data.lead_minutes,
  };
  const res = await api.post('/reminders', payload);
  const r = res.data.reminder || res.data;
  return {
    id: r.id || r.reminder_id || `REM-${Date.now().toString().slice(-6)}`,
    patient_id: r.patient_id || data.patient_id,
    email: r.email || data.email,
    title: r.title || data.title,
    notes: r.notes || data.notes || '',
    appointment_at: r.appointment_at || data.appointment_at,
    lead_minutes: r.lead_minutes || data.lead_minutes,
    status: r.status || 'confirmed',
    calendar_link: r.calendar_link || r.google_calendar_url || '',
  };
}

export async function deleteReminder(id: string): Promise<void> {
  try {
    await api.delete(`/reminders/${id}`);
  } catch (err) {
    console.warn('Error deleting reminder:', err);
  }
}
