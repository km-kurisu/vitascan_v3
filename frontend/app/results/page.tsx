'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  ClipboardCheck,
  FileText,
  Calendar,
  ChevronDown,
  ChevronUp,
  Utensils,
  Pill,
  Activity,
  Sun,
  Lightbulb,
  Ban,
  CheckCircle2,
  AlertCircle,
  Clock,
  Sparkles,
  Info,
  ChevronRight,
} from 'lucide-react';
import { fetchResults, ModCFrontendOutput, DeficiencyItem } from '@/lib/api';
import { saveScanToSupabase } from '@/lib/supabase';
import { useLanguage } from '@/lib/i18n/LanguageContext';

export default function ResultsPage() {
  const { t } = useLanguage();
  const [data, setData] = useState<ModCFrontendOutput | null>(null);
  const [expandedCards, setExpandedCards] = useState<Record<string, boolean>>({
    b12: true,
    vitamin_d: true,
    iron: true,
  });

  useEffect(() => {
    fetchResults().then((res) => {
      setData(res);
      if (res && res.patient?.patient_id) {
        saveScanToSupabase({
          scan_id: `SCAN-${Date.now()}`,
          patient_id: res.patient.patient_id,
          overall_risk_band: res.summary?.overall_risk_band || 'moderate',
          flagged_count: res.summary?.flagged_deficiency_count || 0,
          mod_c_output: res,
          deficiencies: res.deficiencies,
        }).catch((e) => console.warn('Scan auto-save to Supabase failed:', e));
      }
    });
  }, []);

  const toggleExpand = (id: string) => {
    setExpandedCards((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const bandLabel = (band: string) => {
    const key = band?.toLowerCase();
    if (key === 'severe' || key === 'moderate' || key === 'mild' || key === 'borderline' || key === 'normal') {
      return t(`severity.${key}`);
    }
    return band;
  };

  const getDeficiencyTitle = (item: DeficiencyItem) => {
    const key = `deficiency.${item.type.toLowerCase()}`;
    const translated = t(key);
    if (translated !== key) return translated;
    return item.title || item.type;
  };

  const getDeficiencyExplanation = (item: DeficiencyItem) => {
    const key = `explanation.${item.type.toLowerCase()}`;
    const translated = t(key);
    if (translated !== key) return translated;
    return item.explanation;
  };

  const translateParamName = (name: string) => {
    if (name.includes('Ferritin')) return t('param.ferritin');
    if (name.includes('Hemoglobin')) return t('param.hemoglobin');
    if (name.includes('B12')) return t('param.b12');
    if (name.includes('MCV') || name.includes('Mean Corpuscular')) return t('param.mcv');
    if (name.includes('TIBC') || name.includes('Binding Capacity')) return t('param.tibc');
    if (name.includes('Hematocrit')) return t('param.hematocrit');
    return name;
  };

  const translateStatus = (status: string) => {
    const s = status.toLowerCase();
    if (s === 'low') return t('status.low');
    if (s === 'borderline') return t('status.borderline');
    if (s === 'normal') return t('status.normal');
    if (s === 'high') return t('status.high');
    return status;
  };

  const getFoodRecommendation = (item: DeficiencyItem) => {
    const key = `reco.${item.type.toLowerCase()}.foods`;
    const translated = t(key);
    if (translated !== key) return translated;
    return item.recommendations?.foods || item.diet_recommendations[0]?.suggestion || t('results.fbFoods');
  };

  const getTipOrSunlightOrSupplement = (item: DeficiencyItem) => {
    const keyPrefix = `reco.${item.type.toLowerCase()}`;
    if (item.type === 'vitamin_d') {
      const tr = t(`${keyPrefix}.sunlight`);
      if (tr !== `${keyPrefix}.sunlight`) return tr;
      return item.recommendations?.sunlight || t('results.fbSunlight');
    }
    if (item.type === 'iron') {
      const tr = t(`${keyPrefix}.tips`);
      if (tr !== `${keyPrefix}.tips`) return tr;
      return item.recommendations?.tips || item.diet_recommendations[1]?.suggestion || t('results.fbTips');
    }
    const tr = t(`${keyPrefix}.supplements`);
    if (tr !== `${keyPrefix}.supplements`) return tr;
    return item.recommendations?.supplements || item.diet_recommendations[1]?.suggestion || t('results.fbSupplements');
  };

  const getAvoidOrLifestyleOrSupplement = (item: DeficiencyItem) => {
    const keyPrefix = `reco.${item.type.toLowerCase()}`;
    if (item.type === 'b12') {
      const tr = t(`${keyPrefix}.lifestyle`);
      if (tr !== `${keyPrefix}.lifestyle`) return tr;
      return item.recommendations?.lifestyle || t('results.fbLifestyle');
    }
    if (item.type === 'vitamin_d') {
      const tr = t(`${keyPrefix}.supplements`);
      if (tr !== `${keyPrefix}.supplements`) return tr;
      return item.recommendations?.supplements || t('results.fbVitaminD');
    }
    const tr = t(`${keyPrefix}.avoid`);
    if (tr !== `${keyPrefix}.avoid`) return tr;
    return item.recommendations?.avoid || item.diet_recommendations[2]?.suggestion || t('results.fbAvoid');
  };

  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center py-24 space-y-4">
        <Activity className="w-10 h-10 text-[#1D61E7] animate-spin" />
        <p className="text-slate-600 font-semibold text-[#1D61E7]">
          {t('results.loading')}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-10 py-4 max-w-6xl mx-auto">
      {/* Top Hero Status Banner */}
      <div className="bg-white p-6 sm:p-8 rounded-3xl border border-slate-200/80 shadow-xs flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div className="flex items-center space-x-5">
          <div className="relative">
            <div className="w-16 h-16 bg-purple-100 text-[#7C3AED] rounded-2xl flex items-center justify-center shadow-xs">
              <ClipboardCheck className="w-9 h-9" />
            </div>
            <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center text-white text-xs border-2 border-white">
              ✓
            </div>
          </div>
          <div className="space-y-1">
            <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">
              {t('results.readyTitle')}
            </h1>
            <p className="text-slate-500 font-medium text-sm">
              {t('results.readySubtitle')}
            </p>
          </div>
        </div>

        {/* Uploaded Report Meta Box */}
        <div className="bg-slate-50/80 border border-slate-200/80 p-4 rounded-2xl flex items-center space-x-4 min-w-[260px]">
          <div className="w-10 h-10 bg-purple-100/80 text-[#7C3AED] rounded-xl flex items-center justify-center flex-shrink-0">
            <FileText className="w-5 h-5" />
          </div>
          <div className="space-y-0.5">
            <p className="text-xs font-bold text-slate-800 uppercase tracking-wider">
              {t('results.uploadedReport')}
            </p>
            <p className="text-xs font-semibold text-slate-600 font-mono">
              {data.uploaded_report?.filename || 'CBC_Blood_Report.pdf'}
            </p>
            <p className="text-[11px] text-slate-500 font-medium flex items-center gap-1">
              <Calendar className="w-3 h-3" />
              <span>{data.uploaded_report?.uploaded_at || t('results.recentlyProcessed')}</span>
            </p>
          </div>
        </div>
      </div>

      {/* Section 1: Detected Deficiencies and Severity */}
      <div className="space-y-5">
        <h2 className="text-xl font-bold text-slate-900 tracking-tight">
          {t('results.sectionDeficiencies')}
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {data.deficiencies.map((item) => {
            const isSevere = item.severity.band === 'Severe';
            const isModerate = item.severity.band === 'Moderate';
            const isMild = item.severity.band === 'Mild';

            return (
              <div
                key={item.id || item.type}
                className={`bg-white p-6 rounded-2xl border shadow-xs space-y-4 relative overflow-hidden transition-all ${
                  isSevere
                    ? 'border-red-200 hover:border-red-300'
                    : isModerate
                    ? 'border-orange-200 hover:border-orange-300'
                    : 'border-amber-200 hover:border-amber-300'
                }`}
              >
                {/* Header row */}
                <div className="flex items-center justify-between">
                  <div>
                    <h3
                      className={`text-lg font-bold ${
                        isSevere
                          ? 'text-red-600'
                          : isModerate
                          ? 'text-orange-600'
                          : 'text-amber-600'
                      }`}
                    >
                      {getDeficiencyTitle(item)}
                    </h3>
                    <p className="text-xs font-semibold text-slate-500">{t('results.deficientBadge')}</p>
                  </div>

                  <span
                    className={`px-3 py-1 text-xs font-bold rounded-full ${
                      isSevere
                        ? 'bg-red-100 text-red-700 border border-red-200'
                        : isModerate
                        ? 'bg-orange-100 text-orange-700 border border-orange-200'
                        : 'bg-amber-100 text-amber-700 border border-amber-200'
                    }`}
                  >
                    {bandLabel(item.severity.band)}
                  </span>
                </div>

                {/* Progress Bar */}
                <div className="space-y-1.5 pt-2">
                  <div className="w-full h-3 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        isSevere
                          ? 'bg-red-500'
                          : isModerate
                          ? 'bg-orange-500'
                          : 'bg-amber-500'
                      }`}
                      style={{ width: item.severity.score_pct }}
                    />
                  </div>
                </div>

                {/* Link to detail GAT reasoning */}
                <div className="pt-2 flex items-center justify-between text-xs">
                  <Link
                    href={`/results/${item.id || item.type}`}
                    className="text-blue-600 hover:text-blue-700 font-bold flex items-center gap-1 group"
                  >
                    <span>{t('results.viewReasoning')}</span>
                    <ChevronRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
                  </Link>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Section 2: Key Blood Parameters Table */}
      <div className="space-y-5">
        <h2 className="text-xl font-bold text-slate-900 tracking-tight">
          {t('results.sectionParams')}
        </h2>

        <div className="bg-white rounded-2xl border border-slate-200/80 shadow-xs overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 border-b border-slate-200/80 text-xs uppercase font-bold text-slate-500 tracking-wider">
                <tr>
                  <th scope="col" className="px-6 py-4">
                    {t('results.thParam')}
                  </th>
                  <th scope="col" className="px-6 py-4">
                    {t('results.thValue')}
                  </th>
                  <th scope="col" className="px-6 py-4">
                    {t('results.thRange')}
                  </th>
                  <th scope="col" className="px-6 py-4 text-center">
                    {t('results.thStatus')}
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-medium text-slate-700">
                {(data.blood_parameters || []).map((param, idx) => (
                  <tr
                    key={idx}
                    className="hover:bg-slate-50/60 transition-colors"
                  >
                    <td className="px-6 py-4 font-bold text-slate-900">
                      {translateParamName(param.name)}
                    </td>
                    <td className="px-6 py-4 font-semibold text-slate-800 font-mono">
                      {param.value}
                    </td>
                    <td className="px-6 py-4 text-slate-500 font-mono">
                      {param.normal_range}
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span
                        className={`inline-block px-3 py-1 text-xs font-bold rounded-full ${
                          param.status === 'Low'
                            ? 'bg-red-100 text-red-600 border border-red-200'
                            : param.status === 'Borderline'
                            ? 'bg-amber-100 text-amber-700 border border-amber-200'
                            : 'bg-emerald-100 text-emerald-700 border border-emerald-200'
                        }`}
                      >
                        {translateStatus(param.status)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Section 3: Detailed Recommendations */}
      <div className="space-y-5">
        <h2 className="text-xl font-bold text-slate-900 tracking-tight">
          {t('results.sectionRecos')}
        </h2>

        <div className="space-y-4">
          {/* Recommendation Cards */}
          {data.deficiencies.map((item) => {
            const cardKey = item.id || item.type;
            const isB12 = cardKey === 'b12';
            const isVitD = cardKey === 'vitamin_d';
            const isIron = cardKey === 'iron';
            const isExpanded = expandedCards[cardKey] ?? true;

            return (
              <div
                key={cardKey}
                className="bg-white rounded-2xl border border-slate-200/80 shadow-xs overflow-hidden transition-all"
              >
                {/* Accordion Header */}
                <div
                  onClick={() => toggleExpand(cardKey)}
                  className="p-6 cursor-pointer flex items-center justify-between hover:bg-slate-50/50 transition-colors"
                >
                  <div className="flex items-center space-x-4">
                    <div
                      className={`w-12 h-12 rounded-xl flex items-center justify-center font-bold text-base shadow-xs ${
                        isB12
                          ? 'bg-red-100 text-red-600'
                          : isVitD
                          ? 'bg-amber-100 text-amber-600'
                          : 'bg-yellow-100 text-yellow-700'
                      }`}
                    >
                      {isB12 ? (
                        <span className="font-extrabold text-sm">B12</span>
                      ) : isVitD ? (
                        <Sun className="w-6 h-6" />
                      ) : (
                        <span className="font-extrabold text-sm">Fe</span>
                      )}
                    </div>
                    <div>
                      <h3 className="text-base font-bold text-slate-900">
                        {getDeficiencyTitle(item)} ({bandLabel(item.severity.band)}{' '}
                        {t('results.deficiencyNoun')})
                      </h3>
                      <p className="text-xs text-slate-500 max-w-xl line-clamp-1">
                        {getDeficiencyExplanation(item)}
                      </p>
                    </div>
                  </div>

                  <button className="text-slate-500 hover:text-slate-600">
                    {isExpanded ? (
                      <ChevronUp className="w-5 h-5" />
                    ) : (
                      <ChevronDown className="w-5 h-5" />
                    )}
                  </button>
                </div>

                {/* Expanded Grid details matching design layout */}
                {isExpanded && (
                  <div className="px-6 pb-6 pt-2 border-t border-slate-100 bg-[#FAFCFF] grid grid-cols-1 md:grid-cols-4 gap-6 text-xs">
                    {/* Column 1: Explanation */}
                    <div className="space-y-2 md:col-span-1 border-r border-slate-100 pr-4">
                      <p className="text-xs text-slate-600 leading-relaxed font-medium">
                        {getDeficiencyExplanation(item)}
                      </p>
                    </div>

                    {/* Column 2: Foods */}
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2 font-bold text-slate-800">
                        <Utensils className="w-4 h-4 text-amber-600" />
                        <span>{t('results.foods')}</span>
                      </div>
                      <p className="text-slate-600 leading-relaxed font-medium">
                        {getFoodRecommendation(item)}
                      </p>
                    </div>

                    {/* Column 3: Supplements / Sunlight / Tips */}
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2 font-bold text-slate-800">
                        {isVitD ? (
                          <>
                            <Sun className="w-4 h-4 text-amber-500" />
                            <span>{t('results.sunlight')}</span>
                          </>
                        ) : isIron ? (
                          <>
                            <Lightbulb className="w-4 h-4 text-yellow-600" />
                            <span>{t('results.tips')}</span>
                          </>
                        ) : (
                          <>
                            <Pill className="w-4 h-4 text-red-500" />
                            <span>{t('results.supplements')}</span>
                          </>
                        )}
                      </div>
                      <p className="text-slate-600 leading-relaxed font-medium">
                        {getTipOrSunlightOrSupplement(item)}
                      </p>
                    </div>

                    {/* Column 4: Lifestyle / Supplements / Avoid */}
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2 font-bold text-slate-800">
                        {isB12 ? (
                          <>
                            <Activity className="w-4 h-4 text-blue-600" />
                            <span>{t('results.lifestyle')}</span>
                          </>
                        ) : isVitD ? (
                          <>
                            <Pill className="w-4 h-4 text-[#7C3AED]" />
                            <span>{t('results.supplements')}</span>
                          </>
                        ) : (
                          <>
                            <Ban className="w-4 h-4 text-red-500" />
                            <span>{t('results.avoid')}</span>
                          </>
                        )}
                      </div>
                      <p className="text-slate-600 leading-relaxed font-medium">
                        {getAvoidOrLifestyleOrSupplement(item)}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Section 4: Bottom Recheck Banner Callout */}
      <div className="bg-[#EFF6FF] border border-blue-200/80 p-6 sm:p-8 rounded-3xl shadow-xs flex flex-col sm:flex-row items-center justify-between gap-6">
        <div className="flex items-center space-x-5">
          <div className="w-14 h-14 bg-white text-[#1D61E7] rounded-2xl flex items-center justify-center flex-shrink-0 shadow-xs">
            <Calendar className="w-8 h-8" />
          </div>
          <div className="space-y-1">
            <h3 className="text-xl font-bold text-slate-900">
              {t('results.recheckTitle')}
            </h3>
            <p className="text-sm text-slate-600 font-medium">
              {t('results.recheckSubtitle')}
            </p>
          </div>
        </div>

        <div className="relative flex-shrink-0">
          <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center text-[#1D61E7] shadow-sm">
            <Info className="w-6 h-6" />
          </div>
        </div>
      </div>
    </div>
  );
}
