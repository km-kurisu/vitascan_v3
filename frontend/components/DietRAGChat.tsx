'use client';

import React, { useState } from 'react';
import { Send, Bot, User, Sparkles } from 'lucide-react';
import { sendDietChatQuery } from '@/lib/api';

export default function DietRAGChat({ deficiencyType = "iron" }: { deficiencyType?: string }) {
  const [messages, setMessages] = useState<Array<{ sender: 'user' | 'bot'; text: string }>>([
    { sender: 'bot', text: `Hello! I am your Groq-powered FSSAI Diet Assistant. Ask me anything about managing ${deficiencyType} deficiency through Indian foods, recipe ideas, or meal timing!` }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = input.trim();
    setMessages(prev => [...prev, { sender: 'user', text: userMsg }]);
    setInput('');
    setLoading(true);

    try {
      const res = await sendDietChatQuery(userMsg, deficiencyType);
      setMessages(prev => [...prev, { sender: 'bot', text: res.answer }]);
    } catch (err) {
      setMessages(prev => [...prev, { sender: 'bot', text: "Sorry, I encountered an error connecting to Groq RAG engine." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-4 flex flex-col h-[450px]">
      <div className="flex items-center space-x-2 border-b border-slate-200 pb-3 mb-3">
        <Sparkles className="w-5 h-5 text-amber-400" />
        <h4 className="font-bold text-white text-base">FSSAI Diet & Nutrition RAG Chat Assistant</h4>
      </div>

      <div className="flex-1 overflow-y-auto space-y-3 pr-2 text-sm">
        {messages.map((m, idx) => (
          <div key={idx} className={`flex items-start space-x-2 ${m.sender === 'user' ? 'justify-end' : ''}`}>
            {m.sender === 'bot' && (
              <div className="w-7 h-7 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center shrink-0">
                <Bot className="w-4 h-4" />
              </div>
            )}
            <div className={`p-3 rounded-2xl max-w-[80%] ${m.sender === 'user' ? 'bg-indigo-600 text-white rounded-tr-none' : 'bg-slate-100 text-slate-200 border border-slate-200 rounded-tl-none'}`}>
              {m.text}
            </div>
            {m.sender === 'user' && (
              <div className="w-7 h-7 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center shrink-0">
                <User className="w-4 h-4" />
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="text-xs text-slate-500 animate-pulse">Groq RAG is analyzing nutritional guidelines...</div>
        )}
      </div>

      <form onSubmit={handleSend} className="mt-3 flex items-center space-x-2 pt-2 border-t border-slate-200">
        <input 
          type="text"
          placeholder="Ask a dietary question (e.g. Can I take iron with milk?)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="flex-1 bg-slate-100 border border-slate-200 rounded-xl px-4 py-2 text-sm text-white focus:outline-none focus:border-indigo-500"
        />
        <button type="submit" disabled={loading} className="p-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white transition disabled:opacity-50">
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
}
