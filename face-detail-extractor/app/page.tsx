"use client";

import { useState, useRef, useEffect, useCallback, DragEvent, ChangeEvent, ClipboardEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, X, Loader2, Search, User, Fingerprint, Sparkles } from "lucide-react";

interface Prediction {
  age: number;
  gender: string;
  race: string;
  confidence: {
    gender_score: number;
    race_score: number;
  };
}

const API_URL = "https://iprq-face-details-model.hf.space/predict";

const RACE_COLORS: Record<string, string> = {
  White: "text-sky-400 bg-sky-400/10 border-sky-400/20",
  Black: "text-amber-400 bg-amber-400/10 border-amber-400/20",
  Asian: "text-emerald-400 bg-emerald-400/10 border-emerald-400/20",
  Indian: "text-orange-400 bg-orange-400/10 border-orange-400/20",
  Others: "text-purple-400 bg-purple-400/10 border-purple-400/20",
};

function ConfidenceBar({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100);
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-[10px] uppercase tracking-wider font-bold text-zinc-500">
        <span>{label}</span>
        <span className="text-zinc-300">{pct}%</span>
      </div>
      <div className="h-1 w-full rounded-full bg-zinc-800 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
          className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-purple-500"
        />
      </div>
    </div>
  );
}

export default function Home() {
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("image/")) {
      setError("Format not supported. Please use an image.");
      return;
    }
    setFile(f);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  }, []);

  useEffect(() => {
    function onGlobalPaste(e: globalThis.ClipboardEvent) {
      const item = Array.from(e.clipboardData?.items ?? []).find((i) =>
        i.type.startsWith("image/")
      );
      if (item) {
        const f = item.getAsFile();
        if (f) handleFile(f);
      }
    }
    window.addEventListener("paste", onGlobalPaste);
    return () => window.removeEventListener("paste", onGlobalPaste);
  }, [handleFile]);

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(API_URL, { method: "POST", body: form });
      if (!res.ok) throw new Error("Our AI is taking a break. Try again later.");
      const data: Prediction = await res.json();
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setPreview(null);
    setFile(null);
    setResult(null);
    setError(null);
  }

  return (
    <div className="min-h-screen bg-black text-zinc-100 selection:bg-indigo-500/30 font-sans selection:text-indigo-200">
      {/* Background Gradients */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-[10%] -left-[10%] w-[40%] h-[40%] bg-indigo-500/10 blur-[120px] rounded-full" />
        <div className="absolute bottom-[10%] right-[10%] w-[30%] h-[30%] bg-purple-500/10 blur-[120px] rounded-full" />
      </div>

      <main className="relative z-10 max-w-2xl mx-auto px-6 py-20">
        {/* Header */}
        <header className="text-center mb-12 space-y-4">
          <motion.h1 
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
            className="text-5xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-b from-white to-zinc-500"
          >
            Face Analyzer
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
            className="text-zinc-500 text-lg max-w-md mx-auto leading-relaxed"
          >
            Deep learning perception for age, gender, and ethnicity.
          </motion.p>
        </header>

        <div className="space-y-6">
          {/* Upload Zone */}
          <motion.div
            layout
            className={`group relative rounded-3xl border transition-all duration-500 overflow-hidden
              ${dragging ? "border-indigo-500 bg-indigo-500/5" : "border-zinc-800 bg-zinc-900/50 hover:border-zinc-700"}
              ${preview ? "p-2" : "p-12"}
            `}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
            onPaste={(e: ClipboardEvent<HTMLDivElement>) => {
              const item = Array.from(e.clipboardData.items).find((i) =>
                i.type.startsWith("image/")
              );
              if (item) { const f = item.getAsFile(); if (f) handleFile(f); }
            }}
            onClick={() => !preview && inputRef.current?.click()}
          >
            <AnimatePresence mode="wait">
              {!preview ? (
                <motion.div 
                  key="upload-prompt"
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="flex flex-col items-center gap-4 cursor-pointer"
                >
                  <div className="w-16 h-16 rounded-2xl bg-zinc-800 flex items-center justify-center group-hover:scale-110 transition-transform duration-500">
                    <Upload className="text-zinc-400 group-hover:text-indigo-400 transition-colors" />
                  </div>
                  <div className="text-center">
                    <p className="text-zinc-200 font-medium">Drop face here</p>
                    <p className="text-zinc-500 text-sm">or click to browse from device</p>
                  </div>
                </motion.div>
              ) : (
                <motion.div key="preview" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="relative">
                  <img src={preview} alt="Preview" className="w-full aspect-[4/3] object-cover rounded-2xl border border-white/5" />
                  <button 
                    onClick={(e) => { e.stopPropagation(); reset(); }}
                    className="absolute top-4 right-4 w-10 h-10 rounded-full bg-black/60 backdrop-blur-md flex items-center justify-center border border-white/10 hover:bg-black/80 transition-colors"
                  >
                    <X size={18} />
                  </button>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />

          {/* Action Button */}
          <button
            onClick={analyze}
            disabled={!file || loading}
            className="group relative w-full h-14 rounded-2xl bg-white text-black font-bold text-sm overflow-hidden transition-all hover:scale-[1.02] active:scale-[0.98] disabled:opacity-20 disabled:grayscale disabled:scale-100"
          >
            {loading ? (
              <Loader2 className="animate-spin mx-auto" size={20} />
            ) : (
              <div className="flex items-center justify-center gap-2">
                <Search size={18} />
                Run Neural Inference
              </div>
            )}
          </button>

          {/* Error Message */}
          {error && (
            <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-xs font-medium text-center">
              {error}
            </motion.div>
          )}

          {/* Results Area */}
          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0, y: 40 }}
                animate={{ opacity: 1, y: 0 }}
                className="grid grid-cols-1 md:grid-cols-2 gap-4"
              >
                {/* Age Block */}
                <div className="p-8 rounded-3xl bg-zinc-900 border border-zinc-800 flex flex-col justify-between">
                  <div className="text-[10px] font-bold uppercase tracking-widest text-zinc-500 flex items-center gap-2">
                    <User size={12} /> Estimated Age
                  </div>
                  <div className="mt-4 flex items-baseline gap-2">
                    <span className="text-7xl font-black text-white">{Math.round(result.age)}</span>
                    <span className="text-zinc-500 font-medium">years</span>
                  </div>
                </div>

                {/* Identity Block */}
                <div className="p-8 rounded-3xl bg-zinc-900 border border-zinc-800 space-y-6">
                  <div className="text-[10px] font-bold uppercase tracking-widest text-zinc-500 flex items-center gap-2">
                    <Fingerprint size={12} /> Identity Traits
                  </div>
                  
                  <div className="flex flex-wrap gap-2">
                    <span className={`px-4 py-1.5 rounded-full text-[11px] font-bold border transition-colors ${
                      result.gender === "Male" ? "text-blue-400 bg-blue-400/10 border-blue-400/20" : "text-pink-400 bg-pink-400/10 border-pink-400/20"
                    }`}>
                      {result.gender.toUpperCase()}
                    </span>
                    <span className={`px-4 py-1.5 rounded-full text-[11px] font-bold border ${RACE_COLORS[result.race]}`}>
                      {result.race.toUpperCase()}
                    </span>
                  </div>

                  <div className="space-y-4 pt-2 border-t border-zinc-800">
                    <ConfidenceBar label="Gender Accuracy" value={result.confidence.gender_score} />
                    <ConfidenceBar label="Race Accuracy" value={result.confidence.race_score} />
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <footer className="mt-20 text-center">
          <p className="text-[10px] font-bold uppercase tracking-[0.3em] text-zinc-700">
            UTKFace Dataset Analysis • 2026 Edition
          </p>
        </footer>
      </main>
    </div>
  );
}