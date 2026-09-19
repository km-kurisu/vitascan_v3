'use client';

import React, { useState, useEffect } from 'react';
import { useUser } from '@clerk/nextjs';
import { Calendar, BellRing, Trash2, Plus, RefreshCw, CheckCircle2 } from 'lucide-react';
import {
  fetchReminders,
  createReminder,
  deleteReminder,
  LEAD_OPTIONS,
  Reminder,
} from '@/lib/reminders';
import { fetchPatientProfileFromSupabase } from '@/lib/supabase';

export default function RemindersPage() {
  const { user, isLoaded } = useUser();
  const [reminders, setReminders] = useState<Reminder[]>([]);
  const [patientId, setPatientId] = useState('');
  const [email, setEmail] = useState('');
  const [title, setTitle] = useState('');
  const [date, setDate] = useState('');
  const [time, setTime] = useState('');
  const [notes, setNotes] = useState('');
  const [leadMinutes, setLeadMinutes] = useState(1440);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    if (!isLoaded) return;
    const fallback = user?.primaryEmailAddress?.emailAddress || '';
    setEmail(fallback);
    if (user?.id) {
      fetchPatientProfileFromSupabase(user.id).then((profile) => {
        if (profile?.patient_id) {
          setPatientId(profile.patient_id);
          localStorage.setItem('vitascan_patient_id', profile.patient_id);
          loadReminders(profile.patient_id);
        } else {
          const localPid = localStorage.getItem('vitascan_patient_id') || 'PAT-DEMO123';
          setPatientId(localPid);
          loadReminders(localPid);
        }
      });
    } else {
      const localPid = localStorage.getItem('vitascan_patient_id') || 'PAT-DEMO123';
      setPatientId(localPid);
      loadReminders(localPid);
    }
  }, [isLoaded, user]);

  async function loadReminders(pid: string) {
    try {
      const list = await fetchReminders(pid);
      setReminders(list);
    } catch (e) {
      console.warn('Could not load reminders', e);
    }
  }

  const refresh = () => {
    if (patientId) loadReminders(patientId);
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setMessage('');
    setError('');
    try {
      const pid =
        patientId ||
        localStorage.getItem('vitascan_patient_id') ||
        `PAT-${Date.now().toString().slice(-8)}`;
      const appointmentAt = new Date(`${date}T${time}:00`);
      const r = await createReminder({
        patient_id: pid,
        email,
        title,
        notes,
        appointment_at: appointmentAt.toISOString(),
        lead_minutes: leadMinutes,
      });
      setMessage(
        `Reminder saved! Check your email for the confirmation and calendar link.`
      );
      setTitle('');
      setNotes('');
      setPatientId(pid);
      setReminders((prev) => [r, ...prev]);
    } catch (err: any) {
      setError(err?.message || 'Could not create reminder. Is the backend running?');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteReminder(id);
      setReminders((prev) => prev.filter((r) => r.id !== id));
    } catch (e) {
      console.warn('Could not delete reminder', e);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 py-8 px-4 sm:px-6">
      <div className="flex items-center space-x-3">
        <div className="w-12 h-12 rounded-2xl bg-blue-50 border border-blue-100 flex items-center justify-center text-[#1D61E7] shadow-sm">
          <BellRing className="w-6 h-6" />
        </div>
        <div>
          <h1 className="text-3xl font-black text-slate-900 tracking-tight">Reminders & Follow-ups</h1>
          <p className="text-slate-500 font-medium text-sm">
            Set a reminder for your doctor's appointment or supplement routine and get it delivered to your inbox + calendar.
          </p>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl text-sm font-semibold">
          {error}
        </div>
      )}
      {message && (
        <div className="bg-emerald-50 border border-emerald-200 text-emerald-800 px-4 py-3 rounded-xl text-sm font-semibold flex items-center space-x-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-600" />
          <span>{message}</span>
        </div>
      )}

      {/* Create form */}
      <form
        onSubmit={handleCreate}
        className="bg-white p-6 sm:p-8 rounded-3xl border border-slate-200 shadow-sm space-y-5"
      >
        <div className="flex items-center space-x-2 text-[#1D61E7]">
          <Plus className="w-5 h-5" />
          <h2 className="text-xl font-bold text-[#1D61E7]">New Appointment or Supplement Reminder</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div className="space-y-1.5 md:col-span-2">
            <label className="block text-xs font-bold text-slate-700">Title</label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              required
              placeholder="e.g. Follow-up with Dr. Rao / Vitamin D3 Supplement"
              className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-100 text-sm text-slate-900 outline-none placeholder:text-slate-400"
            />
          </div>

          <div className="space-y-1.5">
            <label className="block text-xs font-bold text-slate-700">Date</label>
            <input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              required
              className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-100 text-sm text-slate-900 outline-none"
            />
          </div>

          <div className="space-y-1.5">
            <label className="block text-xs font-bold text-slate-700">Time</label>
            <input
              type="time"
              value={time}
              onChange={(e) => setTime(e.target.value)}
              required
              className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-100 text-sm text-slate-900 outline-none"
            />
          </div>

          <div className="space-y-1.5">
            <label className="block text-xs font-bold text-slate-700">Remind me</label>
            <select
              value={leadMinutes}
              onChange={(e) => setLeadMinutes(Number(e.target.value))}
              className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-100 text-sm text-slate-900 outline-none bg-white"
            >
              {LEAD_OPTIONS.map((opt) => (
                <option key={opt.minutes} value={opt.minutes}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-1.5">
            <label className="block text-xs font-bold text-slate-700">Email (optional override)</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Recipient email"
              className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-100 text-sm text-slate-900 outline-none placeholder:text-slate-400"
            />
          </div>

          <div className="space-y-1.5 md:col-span-2">
            <label className="block text-xs font-bold text-slate-700">Notes (optional)</label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={2}
              placeholder="e.g. Bring blood report or take 60,000 IU capsule with milk"
              className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:border-blue-500 focus:ring-2 focus:ring-blue-100 text-sm text-slate-900 outline-none placeholder:text-slate-400"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={saving}
          className="px-8 py-3.5 bg-[#1D61E7] hover:bg-blue-700 disabled:bg-slate-300 text-white font-bold rounded-xl shadow-lg shadow-blue-600/20 transition-all flex items-center space-x-2 text-sm"
        >
          <Calendar className="w-4 h-4" />
          <span>{saving ? 'Saving...' : 'Save Reminder'}</span>
        </button>
      </form>

      {/* Reminders list */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-bold text-slate-900">Your Reminders ({reminders.length})</h2>
          <button
            onClick={refresh}
            className="text-xs font-bold text-[#1D61E7] hover:bg-blue-50 px-3 py-1.5 rounded-lg border border-blue-100 transition-colors flex items-center space-x-1"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Refresh</span>
          </button>
        </div>

        {reminders.length === 0 ? (
          <div className="bg-white p-8 rounded-2xl border border-slate-200 text-center space-y-2">
            <p className="text-slate-500 text-sm font-medium">
              No reminders scheduled yet. Create one above to get automated email and calendar alerts.
            </p>
          </div>
        ) : (
          reminders.map((r) => (
            <div
              key={r.id}
              className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm flex items-start justify-between gap-4 hover:border-slate-300 transition-colors"
            >
              <div className="space-y-1.5">
                <div className="flex items-center space-x-2">
                  <span className="font-bold text-slate-900 text-base">{r.title}</span>
                  <span
                    className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wide ${
                      r.status === 'sent'
                        ? 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                        : 'bg-blue-50 text-blue-700 border border-blue-200'
                    }`}
                  >
                    {r.status === 'sent' ? 'Reminder sent' : 'Confirmed'}
                  </span>
                </div>
                <p className="text-sm text-slate-600 font-medium">
                  {new Date(r.appointment_at).toLocaleString()}
                </p>
                {r.notes && (
                  <p className="text-xs text-slate-500 bg-slate-50 p-2 rounded-lg border border-slate-100">
                    <strong>Notes:</strong> {r.notes}
                  </p>
                )}
                {r.calendar_link && (
                  <a
                    href={r.calendar_link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center space-x-1.5 text-xs font-bold text-[#1D61E7] hover:underline pt-1"
                  >
                    <Calendar className="w-3.5 h-3.5" />
                    <span>Add to Google Calendar</span>
                  </a>
                )}
              </div>
              <button
                onClick={() => handleDelete(r.id)}
                className="text-slate-400 hover:text-red-600 p-2 rounded-lg hover:bg-red-50 transition-colors"
                aria-label="Delete reminder"
              >
                <Trash2 className="w-5 h-5" />
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
