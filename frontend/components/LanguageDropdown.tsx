'use client';

import React from 'react';
import { Globe, ChevronDown } from 'lucide-react';
import { useLanguage } from '@/lib/i18n/LanguageContext';
import { LANGUAGES, Language } from '@/lib/i18n/translations';

export default function LanguageDropdown() {
  const { language, setLanguage, t } = useLanguage();

  return (
    <div className="relative flex items-center">
      <Globe className="w-3.5 h-3.5 text-blue-600 absolute left-2.5 pointer-events-none" />
      <select
        aria-label={t('common.selectLanguage')}
        value={language}
        onChange={(e) => setLanguage(e.target.value as Language)}
        className="appearance-none pl-8 pr-7 py-2 text-xs font-bold text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-xl transition-all cursor-pointer outline-none"
      >
        {LANGUAGES.map((l) => (
          <option key={l.code} value={l.code}>
            {l.label}
          </option>
        ))}
      </select>
      <ChevronDown className="w-3 h-3 text-slate-500 absolute right-2 pointer-events-none" />
    </div>
  );
}
