'use client';

import React, { useState } from 'react';
import {
  Brain,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  CheckCircle2,
  GitMerge,
} from 'lucide-react';
import { ModelGradeDetail } from '@/lib/api';
import { useLanguage } from '@/lib/i18n/LanguageContext';

const TOTAL_BASE = 9;

const CLASS_COLORS: Record<string, string> = {
  'No Anemia': '#10b981',
  IDA: '#f59e0b',
  'B12 Deficiency': '#ef4444',
  'Folate Deficiency': '#7C3AED',
};

const FALLBACK_COLOR = '#1D61E7';

const colorFor = (label: string) => CLASS_COLORS[label] ?? FALLBACK_COLOR;

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

function ProbRow({
  label,
  value,
  isTop,
}: {
  label: string;
  value: number;
  isTop: boolean;
}) {
  const color = colorFor(label);
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span
          className={`font-bold ${isTop ? 'text-slate-900' : 'text-slate-500'}`}
        >
          {label}
        </span>
        <span
          className={`font-mono font-bold ${isTop ? 'text-slate-900' : 'text-slate-500'}`}
        >
          {pct(value)}
        </span>
      </div>
      <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${Math.max(value * 100, value > 0 ? 1 : 0)}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

export default function ModelVerdictCard({
  model,
  modelConfidence,
}: {
  model: ModelGradeDetail | null | undefined;
  modelConfidence?: number | null;
}) {
  const { t } = useLanguage();
  const [showBlend, setShowBlend] = useState(false);

  if (!model) return null;

  const confidence = modelConfidence ?? Math.max(...Object.values(model.proba ?? {}));
  const topClass =
    Object.entries(model.proba ?? {}).sort((a, b) => b[1] - a[1])[0]?.[0] ??
    model.etiology;
  const noAnemia = model.decision_code === 0;
  const accent = noAnemia ? '#10b981' : colorFor(model.etiology);

  const missing = model.na_count ?? 0;
  const complete = model.complete_input ?? missing === 0;

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-xs overflow-hidden">
      {/* Header */}
      <div className="p-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center space-x-4">
          <div
            className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0"
            style={{ backgroundColor: `${accent}1a`, color: accent }}
          >
            <Brain className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-900 tracking-tight">
              {t('results.modelVerdictTitle')}
            </h2>
            <p className="text-xs text-slate-500 font-medium">
              {t('results.modelVerdictSubtitle')}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span
            className="px-3 py-1.5 text-xs font-bold rounded-full border"
            style={{
              backgroundColor: `${accent}14`,
              color: accent,
              borderColor: `${accent}33`,
            }}
          >
            {t('results.modelConfidence')} {pct(confidence)}
          </span>
          <span className="text-[11px] font-mono text-slate-400">
            #{model.decision_code}
          </span>
        </div>
      </div>

      {/* Verdict row */}
      <div className="px-6 pb-6">
        <div className="bg-slate-50/80 border border-slate-200/80 rounded-2xl p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            {noAnemia ? (
              <CheckCircle2 className="w-6 h-6 text-emerald-500 flex-shrink-0" />
            ) : (
              <AlertTriangle className="w-6 h-6 flex-shrink-0" style={{ color: accent }} />
            )}
            <div>
              <p className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                {t('results.modelEtiology')}
              </p>
              <p className="text-xl font-extrabold" style={{ color: accent }}>
                {model.etiology}
              </p>
            </div>
          </div>

          <div className="text-left sm:text-right">
            <p className="text-xs font-bold text-slate-500 uppercase tracking-wider">
              {t('results.modelTopClass')}
            </p>
            <p className="text-sm font-bold text-slate-800">{topClass}</p>
          </div>
        </div>
      </div>

      {/* Blended probability breakdown */}
      <div className="px-6 pb-6 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-slate-800">
            {t('results.modelProbabilities')}
          </h3>
          <button
            onClick={() => setShowBlend((v) => !v)}
            className="inline-flex items-center gap-1 text-xs font-bold text-blue-600 hover:text-blue-700"
          >
            <GitMerge className="w-3.5 h-3.5" />
            <span>{t('results.modelBlendToggle')}</span>
            {showBlend ? (
              <ChevronUp className="w-3.5 h-3.5" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5" />
            )}
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-4">
          {Object.entries(model.proba ?? {}).map(([label, value]) => (
            <ProbRow
              key={label}
              label={label}
              value={value}
              isTop={label === topClass}
            />
          ))}
        </div>

        {showBlend && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-slate-100">
            <div className="space-y-3">
              <p className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                {t('results.modelXgbCascade')}
              </p>
              {Object.entries(model.proba_xgb ?? {}).map(([label, value]) => (
                <ProbRow
                  key={`xgb-${label}`}
                  label={label}
                  value={value}
                  isTop={label === topClass}
                />
              ))}
            </div>
            <div className="space-y-3">
              <p className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                {t('results.modelGnn')}
              </p>
              {Object.entries(model.proba_gnn ?? {}).map(([label, value]) => (
                <ProbRow
                  key={`gnn-${label}`}
                  label={label}
                  value={value}
                  isTop={label === topClass}
                />
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Data completeness footer */}
      <div className="px-6 py-4 bg-slate-50/60 border-t border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs">
        <span className="font-semibold text-slate-600">
          {t('results.modelBiomarkersUsed')}{' '}
          <span className="font-mono">
            {TOTAL_BASE - missing}/{TOTAL_BASE}
          </span>
        </span>
        <span
          className={`inline-flex items-center gap-1.5 font-bold ${
            complete ? 'text-emerald-600' : 'text-amber-600'
          }`}
        >
          {complete ? (
            <CheckCircle2 className="w-3.5 h-3.5" />
          ) : (
            <AlertTriangle className="w-3.5 h-3.5" />
          )}
          {complete
            ? t('results.modelComplete')
            : t('results.modelImputed').replace('{n}', String(missing))}
        </span>
      </div>
    </div>
  );
}
