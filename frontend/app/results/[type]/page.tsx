'use client';

import React, { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  ArrowLeft,
  Brain,
  Eye,
  ShieldCheck,
  CheckCircle2,
  BarChart3,
  Utensils,
  Pill,
  Activity,
} from 'lucide-react';
import { fetchResults, ModCFrontendOutput, DeficiencyItem } from '@/lib/api';
import { useLanguage } from '@/lib/i18n/LanguageContext';

export default function DeficiencyDetailPage() {
  const params = useParams();
  const deficiencyType = (params.type as string) || 'iron';
  const { t } = useLanguage();

  const bandLabel = (band: string) => {
    const key = band?.toLowerCase();
    if (key === 'severe' || key === 'moderate' || key === 'mild' || key === 'borderline' || key === 'normal') {
      return t(`severity.${key}`);
    }
    return band;
  };

  const getDeficiencyTitle = (d: DeficiencyItem) => {
    const key = `deficiency.${d.type.toLowerCase()}`;
    const translated = t(key);
    if (translated !== key) return translated;
    return d.title || d.type;
  };

  const getDeficiencyExplanation = (d: DeficiencyItem) => {
    const key = `explanation.${d.type.toLowerCase()}`;
    const translated = t(key);
    if (translated !== key) return translated;
    return d.explanation;
  };

  const translateBiomarkerName = (name: string) => {
    if (name.includes('Ferritin')) return t('param.ferritin');
    if (name.includes('Hemoglobin')) return t('param.hemoglobin');
    if (name.includes('B12')) return t('param.b12');
    if (name.includes('MCV') || name.includes('Mean Corpuscular')) return t('param.mcv');
    if (name.includes('TIBC') || name.includes('Binding Capacity')) return t('param.tibc');
    if (name.includes('Hematocrit')) return t('param.hematocrit');
    return name;
  };

  const translateSourceRegion = (source: string) => {
    const s = source.toLowerCase();
    if (s.includes('eye')) return t('source.eyes');
    if (s.includes('tongue')) return t('source.tongue');
    if (s.includes('skin') || s.includes('nail')) return t('source.skin');
    if (s.includes('none')) return t('source.none');
    return source;
  };

  const [data, setData] = useState<ModCFrontendOutput | null>(null);
  const [detail, setDetail] = useState<DeficiencyItem | null>(null);

  useEffect(() => {
    fetchResults().then((res) => {
      setData(res);
      const found = res.deficiencies.find(
        (d) =>
          d.id?.toLowerCase() === deficiencyType.toLowerCase() ||
          d.type.toLowerCase() === deficiencyType.toLowerCase()
      );
      if (found) {
        setDetail(found);
      }
    });
  }, [deficiencyType]);

  if (!data || !detail) {
    return (
      <div className="flex flex-col items-center justify-center py-20 space-y-4">
        <Activity className="w-8 h-8 text-blue-600 animate-spin" />
        <div className="text-slate-600 font-medium">
          {t('detail.loading')}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8 py-2 max-w-4xl mx-auto">
      {/* Back Button */}
      <Link
        href="/results"
        className="inline-flex items-center space-x-2 text-sm font-semibold text-slate-600 hover:text-blue-600 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        <span>{t('detail.back')}</span>
      </Link>

      {/* Title Header Card */}
      <div className="bg-white p-6 sm:p-8 rounded-3xl border border-slate-200/80 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <span className="text-xs font-bold text-blue-600 uppercase tracking-widest">
              {t('detail.biomarkerHeading')}
            </span>
            <h1 className="text-3xl font-extrabold text-slate-900 capitalize">
              {getDeficiencyTitle(detail)} {t('detail.detailSuffix')}
            </h1>
          </div>

          <div
            className={`px-4 py-2 text-sm font-extrabold uppercase rounded-2xl text-white shadow-xs ${
              detail.severity.band === 'Severe'
                ? 'bg-red-500'
                : detail.severity.band === 'Moderate'
                ? 'bg-orange-500'
                : 'bg-amber-500'
            }`}
          >
            {bandLabel(detail.severity.band)} {t('detail.riskWord')} ({detail.severity.score_pct})
          </div>
        </div>

        {/* LLM Plain English Narrative Card */}
        <div className="p-5 bg-blue-50/60 border border-blue-200/80 rounded-2xl space-y-2">
          <div className="flex items-center space-x-2 text-blue-900 font-bold text-sm">
            <Brain className="w-5 h-5 text-blue-600" />
            <span>{t('detail.llmHeading')}</span>
          </div>
          <p className="text-slate-800 text-sm leading-relaxed font-medium">
            "{getDeficiencyExplanation(detail)}"
          </p>
        </div>
      </div>

      {/* GAT Key Contributors Attention Weights Section */}
      <div className="bg-white p-6 sm:p-8 rounded-3xl border border-slate-200/80 shadow-xs space-y-6">
        <div className="flex items-center space-x-3 text-slate-900 font-bold text-lg">
          <BarChart3 className="w-6 h-6 text-blue-600" />
          <span>{t('detail.gatHeading')}</span>
        </div>

        <div className="space-y-4">
          {detail.key_contributors.map((contrib) => (
            <div key={contrib.biomarker} className="space-y-1.5">
              <div className="flex justify-between text-xs font-semibold text-slate-700">
                <span>{translateBiomarkerName(contrib.biomarker)}</span>
                <span className="text-blue-600 font-bold">
                  {contrib.impact_pct}% {t('detail.gatWeightWord')}
                </span>
              </div>
              <div className="w-full bg-slate-100 rounded-full h-3 overflow-hidden">
                <div
                  className="bg-blue-600 h-3 rounded-full transition-all duration-500"
                  style={{ width: `${contrib.impact_pct}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Path A Cross-check & Baseline Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Path A Cross-check */}
        <div className="bg-white p-6 rounded-3xl border border-slate-200/80 shadow-xs space-y-4">
          <div className="flex items-center space-x-2 text-slate-900 font-bold text-base">
            <Eye className="w-5 h-5 text-emerald-600" />
            <span>{t('detail.pathAHeading')}</span>
          </div>

          <div className="p-4 bg-slate-50 rounded-2xl border border-slate-200/80 text-xs space-y-2.5">
            <div className="flex justify-between">
              <span className="text-slate-500">{t('detail.crosscheckAvailable')}</span>
              <span className="font-bold text-slate-800">
                {detail.crosscheck.available ? t('detail.yes') : t('detail.no')}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">{t('detail.sourceRegion')}</span>
              <span className="font-bold text-slate-800 capitalize">
                {translateSourceRegion(detail.crosscheck.source)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">{t('detail.agrees')}</span>
              <span className="font-bold text-emerald-600">
                {detail.crosscheck.agrees ? t('detail.confirmed') : t('detail.pending')}
              </span>
            </div>
          </div>
        </div>

        {/* Baseline Benchmark Comparison */}
        <div className="bg-white p-6 rounded-3xl border border-slate-200/80 shadow-xs space-y-4">
          <div className="flex items-center space-x-2 text-slate-900 font-bold text-base">
            <BarChart3 className="w-5 h-5 text-purple-600" />
            <span>{t('detail.benchmarkHeading')}</span>
          </div>

          <div className="p-4 bg-slate-50 rounded-2xl border border-slate-200/80 text-xs space-y-2.5">
            <div className="flex justify-between">
              <span className="text-slate-500">{t('detail.gatPrediction')}</span>
              <span className="font-bold text-blue-600">
                {detail.severity.score_pct}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">{t('detail.logReg')}</span>
              <span className="font-bold text-slate-700">65.0%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">{t('detail.randForest')}</span>
              <span className="font-bold text-slate-700">69.0%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">{t('detail.xgboost')}</span>
              <span className="font-bold text-slate-700">70.0%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
