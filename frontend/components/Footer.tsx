'use client';

import React from 'react';
import Link from 'next/link';
import { useLanguage } from '@/lib/i18n/LanguageContext';

export default function Footer() {
  const { t } = useLanguage();

  return (
    <footer className="bg-white border-t border-slate-200 text-slate-500 py-8 text-xs">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="space-y-1 text-center md:text-left">
          <p className="font-bold text-slate-800 text-sm">
            <span className="text-[#1D61E7]">Vita</span>
            <span className="text-[#EF4444]">Scan</span> — {t('footer.tagline')}
          </p>
          <p className="text-slate-500">{t('footer.college')}</p>
        </div>
        <div className="flex items-center space-x-6 text-slate-500 font-medium">
          <Link href="/upload" className="hover:text-blue-600 transition-colors">
            {t('footer.bloodUpload')}
          </Link>
          <span>•</span>
          <Link href="/upload" className="hover:text-purple-600 transition-colors">
            {t('footer.skinPhoto')}
          </Link>
          <span>•</span>
          <Link href="/results" className="hover:text-emerald-600 transition-colors">
            {t('footer.results')}
          </Link>
        </div>
      </div>
    </footer>
  );
}
