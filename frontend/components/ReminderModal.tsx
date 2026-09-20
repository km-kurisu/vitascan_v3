'use client';

import React, { useState } from 'react';
import { Calendar, CheckCircle2, X } from 'lucide-react';
import { scheduleReminder } from '@/lib/api';

interface ReminderModalProps {
  isOpen: boolean;
  onClose: () => void;
  defaultTitle?: string;
}

export default function ReminderModal({ isOpen, onClose, defaultTitle = "VitaScan Blood Test Follow-up" }: ReminderModalProps) {
  const [email, setEmail] = useState('');
  const [date, setDate] = useState('');
  const [title, setTitle] = useState(defaultTitle);
  const [notes, setNotes] = useState('Follow-up CBC and Ferritin check');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await scheduleReminder({
        patient_id: "PAT-DEMO123",
        email,
        title,
        notes,
        appointment_at: date ? new Date(`${date}T09:00:00`).toISOString() : new Date().toISOString(),
        lead_minutes: 60,
      });
      setResult(res);
    } catch (err) {
      alert("Error scheduling reminder");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="bg-white border border-slate-200 rounded-2xl max-w-md w-full p-6 relative text-slate-900 shadow-2xl">
        <button onClick={onClose} className="absolute top-4 right-4 text-slate-500 hover:text-slate-700">
          <X className="w-5 h-5" />
        </button>

        <h3 className="text-xl font-bold mb-4 flex items-center space-x-2 text-indigo-600">
          <Calendar className="w-6 h-6" /> <span>Schedule Follow-Up Reminder</span>
        </h3>

        {result ? (
          <div className="space-y-4 text-center">
            <div className="w-12 h-12 rounded-full bg-emerald-500/20 text-emerald-600 flex items-center justify-center mx-auto">
              <CheckCircle2 className="w-8 h-8" />
            </div>
            <p className="font-medium text-emerald-700">Reminder Successfully Scheduled!</p>
            <p className="text-sm text-slate-600">An email notification has been sent to <span className="font-semibold text-slate-900">{email}</span>.</p>

<a
              href={(result && (result.calendar_link || result.data?.calendar_link)) || '#'}
              target="_blank"
              rel="noreferrer"
              className="block w-full py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg text-white font-semibold text-sm shadow-lg hover:from-blue-500 hover:to-indigo-500 transition"
            >
              📅 Add to Google Calendar
            </a>

            <button onClick={onClose} className="mt-2 text-sm text-slate-500 hover:text-slate-800">Close</button>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-500 mb-1">Email Address</label>
              <input 
                type="email" 
                required 
                placeholder="patient@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-slate-100 border border-slate-200 rounded-lg px-3 py-2 text-sm text-slate-900 focus:outline-none focus:border-indigo-500"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-500 mb-1">Scheduled Date</label>
              <input 
                type="date" 
                required 
                value={date}
                onChange={(e) => setDate(e.target.value)}
                className="w-full bg-slate-100 border border-slate-200 rounded-lg px-3 py-2 text-sm text-slate-900 focus:outline-none focus:border-indigo-500"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-500 mb-1">Reminder Title</label>
              <input 
                type="text" 
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className="w-full bg-slate-100 border border-slate-200 rounded-lg px-3 py-2 text-sm text-slate-900 focus:outline-none focus:border-indigo-500"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-500 mb-1">Notes / Instructions</label>
              <textarea 
                rows={2}
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                className="w-full bg-slate-100 border border-slate-200 rounded-lg px-3 py-2 text-sm text-slate-900 focus:outline-none focus:border-indigo-500"
              />
            </div>

            <button 
              type="submit"
              disabled={loading}
              className="w-full py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm shadow-md transition disabled:opacity-50"
            >
              {loading ? "Scheduling..." : "Send Email & Create Calendar Link"}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
