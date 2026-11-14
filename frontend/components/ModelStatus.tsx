"use client";

import { useEffect, useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

type State = "checking" | "ready" | "waking" | "offline";

/**
 * Pings the backend on page load. This shows whether the models are up and,
 * just as important, wakes a sleeping server so the user's first analysis
 * is faster.
 */
export default function ModelStatus({ compact = false }: { compact?: boolean }) {
  const [state, setState] = useState<State>("checking");

  useEffect(() => {
    let cancelled = false;
    const slow = setTimeout(() => {
      if (!cancelled) setState((s) => (s === "checking" ? "waking" : s));
    }, 2500);

    fetch(`${API_URL}/health`, { cache: "no-store" })
      .then((r) => {
        if (!cancelled) setState(r.ok ? "ready" : "offline");
      })
      .catch(() => {
        if (!cancelled) setState("offline");
      })
      .finally(() => clearTimeout(slow));

    return () => {
      cancelled = true;
      clearTimeout(slow);
    };
  }, []);

  const styles: Record<State, { dot: string; text: string; label: string }> = {
    checking: { dot: "bg-slate-400", text: "text-slate-300", label: "Checking models" },
    waking: { dot: "bg-amber-400 animate-pulse", text: "text-amber-200", label: "Waking up models" },
    ready: { dot: "bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.9)]", text: "text-emerald-200", label: "Models online" },
    offline: { dot: "bg-rose-400", text: "text-rose-200", label: "Models offline" },
  };
  const s = styles[state];

  return (
    <span
      role="status"
      aria-live="polite"
      className={`inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] ${
        compact ? "px-2.5 py-1 text-xs" : "px-3 py-1.5 text-sm"
      } font-medium ${s.text}`}
    >
      <span className={`h-2 w-2 rounded-full ${s.dot}`} />
      {s.label}
    </span>
  );
}
