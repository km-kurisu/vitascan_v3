'use client';

import React, { useState } from 'react';
import { Utensils, Sparkles, CalendarDays } from 'lucide-react';
import DietRAGChat from '@/components/DietRAGChat';
import { generateDietPlan } from '@/lib/api';

const ALLERGY_OPTIONS = [
  { id: 'gluten', label: 'Gluten' },
  { id: 'dairy', label: 'Dairy' },
  { id: 'nuts', label: 'Nuts' },
  { id: 'peanut', label: 'Peanut' },
  { id: 'sesame', label: 'Sesame' },
  { id: 'soy', label: 'Soy' },
  { id: 'egg', label: 'Egg' },
  { id: 'fish', label: 'Fish' },
  { id: 'shellfish', label: 'Shellfish' },
];

const DISORDER_OPTIONS = [
  { id: 'lactose_intolerance', label: 'Lactose Intolerance' },
  { id: 'celiac', label: 'Celiac / Gluten Intolerance' },
  { id: 'ibs', label: 'Irritable Bowel Syndrome (IBS)' },
  { id: 'gerd', label: 'Acidity / GERD' },
  { id: 'diabetes', label: 'Diabetes / Prediabetes' },
  { id: 'hypertension', label: 'Hypertension' },
  { id: 'ckd', label: 'Chronic Kidney Disease (CKD)' },
  { id: 'gout', label: 'Gout / High Uric Acid' },
  { id: 'pcos', label: 'PCOS' },
];

const SLOT_ICONS: Record<string, string> = {
  Breakfast: '🌅',
  Lunch: '☀️',
  Snack: '🍪',
  Dinner: '🌙',
};

export default function DietPlannerPage() {
  const [deficiency, setDeficiency] = useState('iron');
  const [dietPref, setDietPref] = useState('vegetarian');
  const [allergies, setAllergies] = useState<string[]>([]);
  const [disorders, setDisorders] = useState<string[]>([]);
  const [plan, setPlan] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const toggle = (list: string[], setList: React.Dispatch<React.SetStateAction<string[]>>, value: string) => {
    setList(list.includes(value) ? list.filter((v) => v !== value) : [...list, value]);
  };

  const handleGenerate = async () => {
    setLoading(true);
    setPlan(null);
    try {
      const res = await generateDietPlan('PAT-DEMO123', deficiency, dietPref, allergies, disorders);
      setPlan(res.data);
    } catch (err) {
      alert('Diet generation failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-10 space-y-8">
      <div className="glass-card p-8 rounded-2xl border border-slate-200 space-y-2">
        <h1 className="text-3xl font-extrabold text-slate-900 flex items-center space-x-3">
          <Utensils className="w-8 h-8 text-emerald-400" /> <span>FSSAI Diet Recommendation & RAG System</span>
        </h1>
        <p className="text-slate-500 text-sm">
          Personalized 7-day Indian meal plans following ICMR-NIN RDA guidelines, grounded in the IFCT2017 knowledge base.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="glass-card p-6 rounded-2xl border border-slate-200 space-y-6">
          <h3 className="font-bold text-slate-900 text-lg">Generate 7-Day Indian Meal Plan</h3>

          <div className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-500 mb-1">Target Deficiency</label>
              <select
                value={deficiency}
                onChange={(e) => setDeficiency(e.target.value)}
                className="w-full bg-white border border-slate-200 rounded-xl px-4 py-3 text-slate-900"
              >
                <option value="iron">Iron Deficiency Anemia</option>
                <option value="anemia">Nutritional Anemia (Mixed)</option>
                <option value="b12">Vitamin B12 Deficiency</option>
                <option value="folate">Folate Deficiency</option>
                <option value="vitamin_d">Vitamin D Deficiency</option>
                <option value="calcium">Calcium Deficiency</option>
                <option value="zinc">Zinc Deficiency</option>
                <option value="protein">Protein Deficiency</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-500 mb-1">Dietary Preference</label>
              <select
                value={dietPref}
                onChange={(e) => setDietPref(e.target.value)}
                className="w-full bg-white border border-slate-200 rounded-xl px-4 py-3 text-slate-900"
              >
                <option value="vegetarian">Vegetarian</option>
                <option value="fishetarian">Fishetarian (Pescatarian)</option>
                <option value="eggetarian">Eggetarian</option>
                <option value="non-vegetarian">Non-Vegetarian</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-500 mb-2">Allergies (foods excluded)</label>
              <div className="flex flex-wrap gap-2">
                {ALLERGY_OPTIONS.map((a) => (
                  <button
                    key={a.id}
                    type="button"
                    onClick={() => toggle(allergies, setAllergies, a.id)}
                    className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition ${
                      allergies.includes(a.id)
                        ? 'bg-red-100 text-red-700 border-red-300'
                        : 'bg-white text-slate-500 border-slate-200 hover:border-red-300'
                    }`}
                  >
                    {a.label}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-500 mb-2">Health Conditions / Disorders</label>
              <div className="flex flex-wrap gap-2">
                {DISORDER_OPTIONS.map((d) => (
                  <button
                    key={d.id}
                    type="button"
                    onClick={() => toggle(disorders, setDisorders, d.id)}
                    className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition ${
                      disorders.includes(d.id)
                        ? 'bg-amber-100 text-amber-700 border-amber-300'
                        : 'bg-white text-slate-500 border-slate-200 hover:border-amber-300'
                    }`}
                  >
                    {d.label}
                  </button>
                ))}
              </div>
            </div>

            <button
              onClick={handleGenerate}
              disabled={loading}
              className="w-full py-3 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-500 hover:from-emerald-500 hover:to-teal-400 text-white font-bold text-sm shadow-lg transition disabled:opacity-60"
            >
              {loading ? 'Generating Plan...' : 'Generate Customized Meal Plan'}
            </button>
          </div>

          {plan && (
            <div className="space-y-3 pt-4 border-t border-slate-200">
              <div className="flex items-center space-x-2 text-emerald-400">
                <CalendarDays className="w-4 h-4" />
                <h4 className="font-bold text-sm">7-Day Meal Structure</h4>
                <span className="ml-auto text-[10px] px-2 py-0.5 rounded-full bg-slate-100 text-slate-500 uppercase">
                  {plan.source || 'curated_kb'}
                </span>
              </div>

              {plan.meal_plan && Object.entries(plan.meal_plan).map(([day, meals]: any) => (
                <div key={day} className="bg-white/80 p-4 rounded-xl text-xs space-y-1.5 text-slate-600">
                  <p className="font-bold text-emerald-600">{day}</p>
                  {['Breakfast', 'Lunch', 'Snack', 'Dinner'].map((slot) => (
                    <p key={slot}>
                      <strong className="text-slate-800">
                        {SLOT_ICONS[slot] || ''} {slot}:
                      </strong>{' '}
                      {meals?.[slot] || '—'}
                    </p>
                  ))}
                </div>
              ))}

              {plan.bioavailability_tips?.length > 0 && (
                <div className="bg-emerald-50 border border-emerald-200 p-4 rounded-xl text-xs space-y-1">
                  <p className="font-bold text-emerald-700">Nutrition Bioavailability Tips</p>
                  {plan.bioavailability_tips.map((tip: string, i: number) => (
                    <p key={i} className="text-emerald-800">• {tip}</p>
                  ))}
                </div>
              )}

              {plan.meal_plan && Object.keys(plan.meal_plan).length > 0 ? (
                <p className="text-[10px] text-slate-400 flex items-center space-x-1">
                  <Sparkles className="w-3 h-3 text-amber-500" /> {plan.source === 'groq_llm_grounded' ? 'Grounded in curated IFCT2017 data' : 'Curated Indian food database (IFCT2017)'}.
                </p>
              ) : (
                <p className="text-xs text-slate-500">
                  {plan.bioavailability_tips?.[0] || 'No safe foods matched this combination. Please consult a physician/dietitian.'}
                </p>
              )}
            </div>
          )}
        </div>

        <DietRAGChat
          deficiencyType={deficiency}
          dietPref={dietPref}
          allergies={allergies}
          disorders={disorders}
        />
      </div>
    </div>
  );
}