'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Activity, LogIn, Menu, X } from 'lucide-react';
import { SignedIn, SignedOut, UserButton } from '@clerk/nextjs';
import LanguageDropdown from '@/components/LanguageDropdown';
import { useLanguage } from '@/lib/i18n/LanguageContext';

export default function Navbar() {
  const pathname = usePathname();
  const { t } = useLanguage();
  const [menuOpen, setMenuOpen] = useState(false);

  const navLinks = [
    { name: t('nav.home') || 'Home', href: '/' },
    { name: t('nav.upload') || 'Upload Scan', href: '/upload' },
    { name: t('nav.results') || 'Results', href: '/results' },
    { name: t('nav.dietPlanner') || 'Diet Planner', href: '/diet-planner' },
    { name: t('nav.reminders') || 'Reminders', href: '/reminders' },
  ];

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center space-x-2.5 group">
          <div className="w-9 h-9 bg-gradient-to-tr from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center text-white shadow-md shadow-blue-500/20 group-hover:scale-105 transition-transform">
            <Activity className="w-5 h-5" />
          </div>
          <div className="flex flex-col">
            <span className="text-xl font-black tracking-tight text-slate-900">
              <span className="text-[#1D61E7]">Vita</span>
              <span className="text-[#EF4444]">Scan</span>
              <span className="text-xs font-semibold text-blue-600 ml-1 bg-blue-50 px-1.5 py-0.5 rounded-full border border-blue-200">V3</span>
            </span>
          </div>
        </Link>

        {/* Navigation Links */}
        <nav className="hidden md:flex items-center space-x-6 text-sm font-medium">
          {navLinks.map((link) => {
            const isActive =
              pathname === link.href ||
              (link.href !== '/' && pathname?.startsWith(link.href));
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`relative py-1.5 transition-colors ${
                  isActive
                    ? 'text-[#1D61E7] font-bold'
                    : 'text-slate-600 hover:text-[#1D61E7]'
                }`}
              >
                {link.name}
                {isActive && (
                  <span className="absolute bottom-0 left-0 w-full h-0.5 bg-[#1D61E7] rounded-full" />
                )}
              </Link>
            );
          })}
        </nav>

        {/* Right Actions & Auth */}
        <div className="flex items-center space-x-3">
          <LanguageDropdown />

          <button
            onClick={() => setMenuOpen((v) => !v)}
            className="md:hidden p-2 text-slate-600 hover:text-[#1D61E7] transition-colors"
            aria-label="Toggle navigation menu"
          >
            {menuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>

          <SignedIn>
            <UserButton afterSignOutUrl="/" />
          </SignedIn>

          <SignedOut>
            <Link
              href="/login"
              className="px-3.5 py-1.5 text-xs font-bold text-white bg-[#1D61E7] hover:bg-blue-700 rounded-lg shadow-sm transition-all flex items-center space-x-1"
            >
              <LogIn className="w-3.5 h-3.5" />
              <span>Sign In</span>
            </Link>
          </SignedOut>
        </div>
      </div>

      {/* Mobile menu */}
      {menuOpen && (
        <nav className="md:hidden border-t border-slate-200 bg-white px-4 py-3 space-y-1">
          {navLinks.map((link) => {
            const isActive =
              pathname === link.href ||
              (link.href !== '/' && pathname?.startsWith(link.href));
            return (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setMenuOpen(false)}
                className={`block py-2.5 px-3 rounded-xl text-sm font-semibold transition-colors ${
                  isActive
                    ? 'bg-blue-50 text-[#1D61E7]'
                    : 'text-slate-700 hover:bg-slate-50'
                }`}
              >
                {link.name}
              </Link>
            );
          })}
        </nav>
      )}
    </header>
  );
}
