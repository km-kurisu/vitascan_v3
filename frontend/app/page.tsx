'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  FileText,
  UploadCloud,
  Sparkles,
  Camera,
  Brain,
  BarChart3,
  Utensils,
  ShieldCheck,
  ArrowRight,
  CheckCircle,
} from 'lucide-react';
import { useUser } from '@clerk/nextjs';
import { uploadBloodReport, uploadSymptomPhoto } from '@/lib/api';
import { useLanguage } from '@/lib/i18n/LanguageContext';

export default function Home() {
  const router = useRouter();
  const { user, isLoaded } = useUser();
  const { t } = useLanguage();

  const [reportFile, setReportFile] = useState<File | null>(null);
  const [symptomFile, setSymptomFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);

  const getActivePatientId = () => {
    if (typeof window !== 'undefined' && localStorage.getItem('vitascan_patient_id')) {
      return localStorage.getItem('vitascan_patient_id')!;
    }
    return user?.id
      ? `PAT-${user.id.replace(/^user_/, '').slice(0, 8).toUpperCase()}`
      : 'PAT-DEFAULT';
  };

  const handleReportUpload = async (file: File) => {
    setLoading(true);
    try {
      await uploadBloodReport(file, getActivePatientId());
      router.push('/results');
    } catch (e) {
      // Fallback redirect to results demo
      router.push('/results');
    } finally {
      setLoading(false);
    }
  };

  const handleSymptomUpload = async (file: File) => {
    setLoading(true);
    try {
      await uploadSymptomPhoto(file, 'skin', getActivePatientId());
      router.push('/results');
    } catch (e) {
      router.push('/results');
    } finally {
      setLoading(false);
    }
  };

  const greetingName = isLoaded && user?.firstName ? user.firstName : null;

  return (
    <div className="space-y-16 py-4 max-w-6xl mx-auto">
      {/* Top Greeting Section */}
      <div className="bg-gradient-to-r from-blue-50/50 via-slate-50 to-indigo-50/40 p-8 rounded-3xl border border-slate-200/60 shadow-xs space-y-2">
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight flex items-center gap-3">
          <span>{greetingName ? t('home.welcomeBack', { name: greetingName }) : t('home.welcome')}</span>
          <span className="text-3xl animate-bounce">👋</span>
        </h1>
        <p className="text-slate-600 font-medium text-base">
          {t('home.subtitle')}
        </p>
      </div>

      {/* Dual Analysis Upload Cards Section */}
      <div className="bg-[#F8FAFC] p-6 sm:p-8 rounded-3xl border border-slate-200/80 shadow-xs grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Card 1: Medical Report Analysis (Blue Theme) */}
        <div className="bg-white p-6 sm:p-8 rounded-2xl border border-blue-100 shadow-sm hover:shadow-md transition-shadow space-y-6 flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className="w-14 h-14 bg-blue-100/70 text-[#1D61E7] rounded-2xl flex items-center justify-center flex-shrink-0 shadow-xs">
                <FileText className="w-7 h-7" />
              </div>
              <div className="space-y-1">
                <h2 className="text-xl font-bold text-slate-900">
                  {t('home.reportTitle')}
                </h2>
                <p className="text-sm text-slate-500 leading-relaxed">
                  {t('home.reportDesc')}
                </p>
              </div>
            </div>

            {/* Drag and Drop Zone */}
            <div className="border-2 border-dashed border-blue-200 hover:border-blue-500 bg-blue-50/30 rounded-2xl p-6 text-center transition-colors relative group cursor-pointer">
              <input
                type="file"
                accept=".pdf,.png,.jpg,.jpeg"
                onChange={(e) => {
                  if (e.target.files?.[0]) {
                    setReportFile(e.target.files[0]);
                    handleReportUpload(e.target.files[0]);
                  }
                }}
                className="absolute inset-0 opacity-0 cursor-pointer w-full h-full z-10"
              />
              <div className="space-y-3">
                <div className="w-12 h-12 bg-white text-[#1D61E7] rounded-full mx-auto flex items-center justify-center shadow-xs group-hover:scale-110 transition-transform">
                  <UploadCloud className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-sm font-bold text-slate-800">
                    {reportFile ? reportFile.name : t('home.reportDropzone')}
                  </p>
                  <p className="text-xs text-slate-500 font-medium mt-0.5">
                    {t('home.reportSupports')}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <button
            onClick={() => {
              if (reportFile) handleReportUpload(reportFile);
              else router.push('/upload');
            }}
            disabled={loading}
            className="w-full py-3.5 px-6 bg-[#1D61E7] hover:bg-blue-700 text-white font-bold rounded-xl shadow-md shadow-blue-600/20 transition-all flex items-center justify-center space-x-2 text-sm"
          >
            <UploadCloud className="w-5 h-5" />
            <span>{reportFile ? t('home.reportAnalyzeBtn') : t('home.reportUploadBtn')}</span>
          </button>
        </div>

        {/* Card 2: Skin Condition Analysis (Purple Theme) */}
        <div className="bg-white p-6 sm:p-8 rounded-2xl border border-purple-100 shadow-sm hover:shadow-md transition-shadow space-y-6 flex flex-col justify-between">
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className="w-14 h-14 bg-purple-100/70 text-[#7C3AED] rounded-2xl flex items-center justify-center flex-shrink-0 shadow-xs">
                <Camera className="w-7 h-7" />
              </div>
              <div className="space-y-1">
                <h2 className="text-xl font-bold text-slate-900">
                  {t('home.skinTitle')}
                </h2>
                <p className="text-sm text-slate-500 leading-relaxed">
                  {t('home.skinDesc')}
                </p>
              </div>
            </div>

            {/* Drag and Drop Zone */}
            <div className="border-2 border-dashed border-purple-200 hover:border-purple-500 bg-purple-50/30 rounded-2xl p-6 text-center transition-colors relative group cursor-pointer">
              <input
                type="file"
                accept="image/*"
                onChange={(e) => {
                  if (e.target.files?.[0]) {
                    setSymptomFile(e.target.files[0]);
                    handleSymptomUpload(e.target.files[0]);
                  }
                }}
                className="absolute inset-0 opacity-0 cursor-pointer w-full h-full z-10"
              />
              <div className="space-y-3">
                <div className="w-12 h-12 bg-white text-[#7C3AED] rounded-full mx-auto flex items-center justify-center shadow-xs group-hover:scale-110 transition-transform">
                  <UploadCloud className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-sm font-bold text-slate-800">
                    {symptomFile ? symptomFile.name : t('home.skinDropzone')}
                  </p>
                  <p className="text-xs text-slate-500 font-medium mt-0.5">
                    {t('home.skinSupports')}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <button
            onClick={() => {
              if (symptomFile) handleSymptomUpload(symptomFile);
              else router.push('/upload');
            }}
            disabled={loading}
            className="w-full py-3.5 px-6 bg-[#7C3AED] hover:bg-purple-700 text-white font-bold rounded-xl shadow-md shadow-purple-600/20 transition-all flex items-center justify-center space-x-2 text-sm"
          >
            <UploadCloud className="w-5 h-5" />
            <span>{symptomFile ? t('home.skinAnalyzeBtn') : t('home.skinUploadBtn')}</span>
          </button>
        </div>
      </div>

      {/* About VitaScan Section */}
      <div id="about" className="space-y-10 pt-4">
        <div className="text-center space-y-2">
          <h2 className="text-3xl font-black text-slate-900 tracking-tight">
            <span className="text-[#1D61E7]">{t('home.aboutFirst')} </span>
            <span className="text-[#EF4444]">{t('home.aboutSecond')}</span>
          </h2>
        </div>

        {/* 6 Feature Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Card 1 */}
          <div className="bg-white p-7 rounded-2xl border border-red-100 shadow-sm hover:shadow-md transition-all space-y-4 text-center">
            <div className="w-24 h-24 bg-red-50 rounded-2xl mx-auto flex items-center justify-center p-3">
              <div className="w-16 h-16 bg-red-100/80 rounded-xl flex items-center justify-center text-red-600 shadow-xs">
                <FileText className="w-9 h-9" />
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-lg font-bold text-slate-900">
                {t('feat1.title')}
              </h3>
              <p className="text-xs font-bold text-red-500 uppercase tracking-wide">
                {t('feat1.badge')}
              </p>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">
              {t('feat1.desc')}
            </p>
          </div>

          {/* Card 2 */}
          <div className="bg-white p-7 rounded-2xl border border-blue-100 shadow-sm hover:shadow-md transition-all space-y-4 text-center">
            <div className="w-24 h-24 bg-blue-50 rounded-2xl mx-auto flex items-center justify-center p-3">
              <div className="w-16 h-16 bg-blue-100/80 rounded-xl flex items-center justify-center text-[#1D61E7] shadow-xs">
                <Camera className="w-9 h-9" />
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-lg font-bold text-slate-900">
                {t('feat2.title')}
              </h3>
              <p className="text-xs font-bold text-blue-600 uppercase tracking-wide">
                {t('feat2.badge')}
              </p>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">
              {t('feat2.desc')}
            </p>
          </div>

          {/* Card 3 */}
          <div className="bg-white p-7 rounded-2xl border border-emerald-100 shadow-sm hover:shadow-md transition-all space-y-4 text-center">
            <div className="w-24 h-24 bg-emerald-50 rounded-2xl mx-auto flex items-center justify-center p-3">
              <div className="w-16 h-16 bg-emerald-100/80 rounded-xl flex items-center justify-center text-emerald-600 shadow-xs">
                <Brain className="w-9 h-9" />
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-lg font-bold text-slate-900">
                {t('feat3.title')}
              </h3>
              <p className="text-xs font-bold text-emerald-600 uppercase tracking-wide">
                {t('feat3.badge')}
              </p>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">
              {t('feat3.desc')}
            </p>
          </div>

          {/* Card 4 */}
          <div className="bg-white p-7 rounded-2xl border border-purple-100 shadow-sm hover:shadow-md transition-all space-y-4 text-center">
            <div className="w-24 h-24 bg-purple-50 rounded-2xl mx-auto flex items-center justify-center p-3">
              <div className="w-16 h-16 bg-purple-100/80 rounded-xl flex items-center justify-center text-[#7C3AED] shadow-xs">
                <BarChart3 className="w-9 h-9" />
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-lg font-bold text-slate-900">
                {t('feat4.title')}
              </h3>
              <p className="text-xs font-bold text-purple-600 uppercase tracking-wide">
                {t('feat4.badge')}
              </p>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">
              {t('feat4.desc')}
            </p>
          </div>

          {/* Card 5 */}
          <div className="bg-white p-7 rounded-2xl border border-amber-100 shadow-sm hover:shadow-md transition-all space-y-4 text-center">
            <div className="w-24 h-24 bg-amber-50 rounded-2xl mx-auto flex items-center justify-center p-3">
              <div className="w-16 h-16 bg-amber-100/80 rounded-xl flex items-center justify-center text-amber-600 shadow-xs">
                <Utensils className="w-9 h-9" />
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-lg font-bold text-slate-900">
                {t('feat5.title')}
              </h3>
              <p className="text-xs font-bold text-amber-600 uppercase tracking-wide">
                {t('feat5.badge')}
              </p>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">
              {t('feat5.desc')}
            </p>
          </div>

          {/* Card 6 */}
          <div className="bg-white p-7 rounded-2xl border border-teal-100 shadow-sm hover:shadow-md transition-all space-y-4 text-center">
            <div className="w-24 h-24 bg-teal-50 rounded-2xl mx-auto flex items-center justify-center p-3">
              <div className="w-16 h-16 bg-teal-100/80 rounded-xl flex items-center justify-center text-teal-600 shadow-xs">
                <ShieldCheck className="w-9 h-9" />
              </div>
            </div>
            <div className="space-y-1">
              <h3 className="text-lg font-bold text-slate-900">
                {t('feat6.title')}
              </h3>
              <p className="text-xs font-bold text-teal-600 uppercase tracking-wide">
                {t('feat6.badge')}
              </p>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">
              {t('feat6.desc')}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
