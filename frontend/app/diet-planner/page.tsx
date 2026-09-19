'use client';

import React, { useState } from 'react';
import { Utensils, Sparkles } from 'lucide-react';
import DietRAGChat from '@/components/DietRAGChat';
import { generateDietPlan } from '@/lib/api';

export default function DietPlannerPage() {
  const [deficiency, setDeficiency] = useState('iron');
  const [dietPref, setDietPref] = useState('vegetarian');
  const [plan, setPlan] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const res = await generateDietPlan("PAT-DEMO123", deficiency, dietPref);
      setPlan(res.data);
    } catch (err) {
      alert("Diet generation failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-10 space-y-8">
      <div className="glass-card p-8 rounded-2xl border border-slate-200 space-y-2">
        <h1 className="text-3xl font-extrabold text-white flex items-center space-x-3">
          <Utensils className="w-8 h-8 text-emerald-400" /> <span>FSSAI Diet Recommendation & RAG System</span>
        </h1>
        <p className="text-slate-500 text-sm">Personalized Indian meal plans complying with ICMR-NIN RDA guidelines powered by Groq LLM.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="glass-card p-6 rounded-2xl border border-slate-200 space-y-6">
          <h3 className="font-bold text-white text-lg">Generate 7-Day Indian Meal Plan</h3>

          <div className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-500 mb-1">Target Deficiency</label>
              <select 
                value={deficiency} 
                onChange={(e) => setDeficiency(e.target.value)}
                className="w-full bg-white border border-slate-200 rounded-xl px-4 py-3 text-white"
              >
                <option value="iron">Iron Deficiency Anemia</option>
                <option value="b12">Vitamin B12 Deficiency</option>
                <option value="folate">Folate Deficiency</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-500 mb-1">Dietary Preference</label>
              <select 
                value={dietPref} 
                onChange={(e) => setDietPref(e.target.value)}
                className="w-full bg-white border border-slate-200 rounded-xl px-4 py-3 text-white"
              >
                <option value="vegetarian">Vegetarian</option>
                <option value="non-vegetarian">Non-Vegetarian</option>
                <option value="eggetarian">Eggetarian</option>
              </select>
            </div>

            <button 
              onClick={handleGenerate}
              disabled={loading}
              className="w-full py-3 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-500 hover:from-emerald-500 hover:to-teal-400 text-white font-bold text-sm shadow-lg transition"
            >
              {loading ? "Generating Plan via Groq RAG..." : "Generate Customized Meal Plan"}
            </button>
          </div>

          {plan && (
            <div className="space-y-3 pt-4 border-t border-slate-200">
              <h4 className="font-bold text-emerald-400 text-sm">Suggested Daily Meal Structure:</h4>
              <div className="bg-white/80 p-4 rounded-xl text-xs space-y-2 text-slate-600">
                <p><strong className="text-white">Breakfast:</strong> {plan.meal_plan?.Breakfast || "Rajgira porridge / Palak Paratha"}</p>
                <p><strong className="text-white">Lunch:</strong> {plan.meal_plan?.Lunch || "Dal Tadka, Rice, Beetroot Poriyal"}</p>
                <p><strong className="text-white">Snack:</strong> {plan.meal_plan?.Snack || "Jaggery & Sesame Chikki"}</p>
                <p><strong className="text-white">Dinner:</strong> {plan.meal_plan?.Dinner || "Spinach Paneer & Roti"}</p>
              </div>
            </div>
          )}
        </div>

        <DietRAGChat deficiencyType={deficiency} />
      </div>
    </div>
  );
}
