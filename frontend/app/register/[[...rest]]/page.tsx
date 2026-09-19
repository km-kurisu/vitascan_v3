'use client';

import React from 'react';
import { SignUp } from '@clerk/nextjs';
import { Activity } from 'lucide-react';

export default function RegisterPage() {
  return (
    <div className="min-h-[80vh] flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-md space-y-6">
        <div className="text-center space-y-2">
          <div className="w-12 h-12 bg-blue-600 rounded-2xl flex items-center justify-center text-white mx-auto shadow-lg shadow-blue-500/30">
            <Activity className="w-7 h-7" />
          </div>
          <h1 className="text-2xl font-bold text-slate-900">Create Account</h1>
          <p className="text-sm text-slate-500">Join VitaScan to track deficiencies & diet recommendations</p>
        </div>

        <div className="flex justify-center bg-white p-6 rounded-2xl border border-slate-200 shadow-xl">
          <SignUp 
            routing="path" 
            path="/register" 
            signInUrl="/login"
            appearance={{
              elements: {
                formButtonPrimary: 'bg-blue-600 hover:bg-blue-700 text-white font-medium text-sm py-2 px-4 rounded-lg',
                card: 'shadow-none border-0 p-0',
              }
            }}
          />
        </div>
      </div>
    </div>
  );
}
