"use client";

import { ChangeEvent, DragEvent, useEffect, useState } from "react";
import {
  AlertCircle,
  Bone,
  CheckCircle2,
  Download,
  FileImage,
  Flame,
  Loader2,
  RotateCcw,
  ShieldAlert,
  UploadCloud,
} from "lucide-react";
import ModelStatus from "./ModelStatus";

type BoneResult = {
  study: {
    type: string;
    body_part: string;
    body_part_display_score: string;
    validation_auc: number;
    body_part_validation_accuracy: number;
  };
  body_part_predictions: {
    name: string;
    score: number;
    display_score: string;
  }[];
  abnormality: {
    score: number;
    display_score: string;
    status:
      | "Positive model finding"
      | "Indeterminate"
      | "Negative model finding";
    display_threshold: string;
  };
  interpretation: string;
  gradcam: {
    image_base64: string;
    explanation: string;
  };
  metrics: {
    auc?: number;
    precision?: number;
    sensitivity?: number;
    specificity?: number;
    f1?: number;
    body_part_accuracy?: number;
  };
};

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export default function BoneAnalyzer() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<BoneResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!isAnalyzing) return;
    setElapsed(0);
    const started = Date.now();
    const timer = setInterval(
      () => setElapsed(Math.round((Date.now() - started) / 1000)),
      1000,
    );
    return () => clearInterval(timer);
  }, [isAnalyzing]);

  function chooseFile(file: File) {
    if (!["image/png", "image/jpeg"].includes(file.type)) {
      setError("Please choose a PNG or JPEG image.");
      return;
    }
    if (file.size > 12 * 1024 * 1024) {
      setError("Please choose an image smaller than 12 MB.");
      return;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) chooseFile(file);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) chooseFile(file);
  }

  async function analyzeImage() {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    setError(null);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 300000);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const response = await fetch(`${API_URL}/api/bone/predict`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.detail ?? "Analysis failed.");
      setResult(payload as BoneResult);
    } catch (caughtError) {
      const errorMessage =
        caughtError instanceof Error
          ? caughtError.name === "AbortError"
            ? "Request timed out. Please try again."
            : caughtError.message
          : "Analysis failed.";
      setError(errorMessage);
    } finally {
      clearTimeout(timeout);
      setIsAnalyzing(false);
    }
  }

  async function downloadReport() {
    if (!selectedFile) return;
    setIsDownloading(true);
    setError(null);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 600000);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const response = await fetch(`${API_URL}/api/bone/report`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail ?? "Report download failed.");
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `medvision-${selectedFile.name.replace(/\.[^.]+$/, "")}-bone-report.pdf`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (caughtError) {
      const errorMessage =
        caughtError instanceof Error
          ? caughtError.name === "AbortError"
            ? "Report download timed out. Please try again."
            : caughtError.message
          : "Report download failed.";
      setError(errorMessage);
    } finally {
      clearTimeout(timeout);
      setIsDownloading(false);
    }
  }

  function reset() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  }

  const positive = result?.abnormality.status === "Positive model finding";
  const negative = result?.abnormality.status === "Negative model finding";
  const StatusIcon = negative ? CheckCircle2 : AlertCircle;

  return (
    <section className="mx-auto w-full max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
      <div className="mb-9 max-w-3xl">
        <div className="flex flex-wrap items-center gap-3">
          <p className="text-sm font-semibold uppercase tracking-[0.18em] text-sky-300">
            Bone X-ray Analysis
          </p>
          <ModelStatus compact />
        </div>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl">
          Automatic musculoskeletal X-ray analysis
        </h1>
        <p className="mt-4 text-lg leading-8 text-slate-300">
          Upload a supported radiograph to detect the body region, estimate
          normal-versus-abnormal appearance, and generate Grad-CAM.
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="space-y-5">
          <div className="fade-up rounded-[2rem] border border-white/10 bg-white/[0.03] p-6 shadow-sm">
            <div
              onDragEnter={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              className={`relative flex min-h-72 flex-col overflow-hidden items-center justify-center rounded-2xl border-2 border-dashed p-6 text-center transition ${
                isDragging
                  ? "border-sky-400 bg-sky-400/10"
                  : "border-white/15 bg-white/[0.04]"
              }`}
            >
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Selected bone X-ray"
                  className="max-h-72 w-full rounded-xl object-contain"
                />
              ) : (
                <>
                  <UploadCloud className="h-9 w-9 text-sky-300" />
                  <p className="mt-4 font-medium">Drag and drop a PNG or JPEG</p>
                  <p className="mt-1 text-sm text-slate-400">Maximum size: 12 MB</p>
                </>
              )}
              {previewUrl && isAnalyzing && <div className="scan-line" />}

              <label className="mt-5 inline-flex cursor-pointer items-center gap-2 rounded-xl bg-white px-4 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-slate-200">
                <FileImage className="h-4 w-4" />
                Choose image
                <input
                  type="file"
                  accept="image/png,image/jpeg"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </label>
            </div>

            {selectedFile && (
              <div className="mt-4 flex items-center justify-between rounded-xl bg-white/[0.04] px-4 py-3">
                <p className="truncate text-sm font-medium">{selectedFile.name}</p>
                <button type="button" onClick={reset} aria-label="Reset">
                  <RotateCcw className="h-4 w-4 text-slate-400" />
                </button>
              </div>
            )}

            {error && (
              <div className="mt-4 rounded-xl border border-rose-400/30 bg-rose-500/10 p-4 text-sm text-rose-200">
                {error}
              </div>
            )}

            <button
              type="button"
              onClick={analyzeImage}
              disabled={!selectedFile || isAnalyzing}
              className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-violet-500 px-5 py-3.5 font-semibold text-white shadow-[0_10px_40px_-12px_rgba(56,189,248,0.7)] disabled:opacity-50"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Detecting region and analyzing
                </>
              ) : (
                <>
                  <Bone className="h-5 w-5" />
                  Analyze image
                </>
              )}
            </button>

            {isAnalyzing && (
            <div className="mt-3 rounded-xl border border-violet-400/30 bg-violet-500/10 p-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-violet-100">
                <Loader2 className="h-4 w-4 animate-spin" />
                Analysis in progress
                <span className="ml-auto font-mono text-xs text-violet-200">
                  {elapsed}s
                </span>
              </div>
              <p className="mt-1 text-sm text-violet-200">
                Please keep this tab open. The first request can take a few
                minutes.
              </p>
            </div>
            )}

          </div>

          <div className="flex gap-3 rounded-2xl border border-amber-400/30 bg-amber-500/10 p-4">
            <ShieldAlert className="h-5 w-5 shrink-0 text-amber-300" />
            <p className="text-sm leading-6 text-amber-100">
              Educational research use only. This model does not diagnose
              fractures or specific conditions.
            </p>
          </div>
        </div>

        <div className="space-y-6">
          {!result ? (
            <div className="flex min-h-[520px] items-center justify-center rounded-[2rem] border border-white/10 bg-white/[0.03] p-8 text-center shadow-sm">
              {isAnalyzing ? (
                <div>
                  <Loader2 className="mx-auto h-10 w-10 animate-spin text-sky-300" />
                  <h2 className="mt-5 text-2xl font-semibold">Analyzing image</h2>
                  <p className="mt-2 text-sm text-slate-400">
                    We are processing your X-ray. This may take a few minutes.
                  </p>
                </div>
              ) : (
                <div>
                  <Bone className="mx-auto h-9 w-9 text-sky-300" />
                  <h2 className="mt-5 text-2xl font-semibold">
                    Results will appear here
                  </h2>
                  <p className="mt-2 text-sm text-slate-400">
                    Upload a supported musculoskeletal radiograph to begin.
                  </p>
                </div>
              )}
            </div>
          ) : (
            <>
              <div className="fade-up rounded-[2rem] border border-white/10 bg-white/[0.03] p-6 shadow-sm">
                <p className="text-sm font-semibold uppercase tracking-[0.16em] text-sky-300">
                  Study
                </p>
                <div className="mt-4 flex flex-wrap items-end justify-between gap-5">
                  <div>
                    <h2 className="text-3xl font-semibold">{result.study.type}</h2>
                    <p className="mt-2 text-sm text-slate-400">
                      Body-region confidence {result.study.body_part_display_score}
                    </p>
                  </div>
                  <div className="rounded-2xl bg-white/[0.04] px-4 py-3 text-right">
                    <p className="text-xs text-slate-400">Body-part accuracy</p>
                    <p className="mt-1 text-2xl font-semibold">
                      {(result.study.body_part_validation_accuracy * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                <div className="mt-6 space-y-3">
                  {result.body_part_predictions.slice(0, 3).map((prediction) => (
                    <div key={prediction.name}>
                      <div className="flex justify-between text-sm">
                        <span>{prediction.name}</span>
                        <span>{prediction.display_score}</span>
                      </div>
                      <div className="mt-2 h-2 rounded-full bg-white/10">
                        <div
                          className="bar-grow h-full rounded-full bg-gradient-to-r from-sky-400 to-violet-400"
                          style={{ width: `${Math.max(1, prediction.score * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="fade-up rounded-[2rem] border border-white/10 bg-white/[0.03] p-6 shadow-sm">
                <p className="text-sm text-slate-400">Abnormality output</p>
                <div className="mt-3 flex flex-wrap items-start justify-between gap-5">
                  <div>
                    <p className="text-5xl font-semibold tracking-tight">
                      {result.abnormality.display_score}
                    </p>
                    <p className="mt-2 text-sm text-slate-400">
                      Decision threshold {result.abnormality.display_threshold}
                    </p>
                  </div>
                  <div
                    className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm font-semibold ${
                      positive
                        ? "border-rose-400/30 bg-rose-500/10 text-rose-300"
                        : negative
                        ? "border-emerald-400/30 bg-emerald-500/10 text-emerald-300"
                        : "border-amber-400/30 bg-amber-500/10 text-amber-300"
                    }`}
                  >
                    <StatusIcon className="h-4 w-4" />
                    {result.abnormality.status}
                  </div>
                </div>
              </div>

              <div className="grid gap-4 sm:grid-cols-3">
                {[
                  ["Abnormality AUC", result.metrics?.auc],
                  ["Sensitivity", result.metrics?.sensitivity],
                  ["Specificity", result.metrics?.specificity],
                ].map(([label, value]) => (
                  <div
                    key={String(label)}
                    className="rounded-2xl border border-white/10 bg-white/[0.03] p-5 shadow-sm"
                  >
                    <p className="text-sm text-slate-400">{String(label)}</p>
                    <p className="mt-2 text-2xl font-semibold">
                      {typeof value === "number" ? value.toFixed(3) : "—"}
                    </p>
                  </div>
                ))}
              </div>

              <div className="grid gap-6 xl:grid-cols-2">
                <div className="fade-up rounded-[2rem] border border-white/10 bg-white/[0.03] p-6 shadow-sm">
                  <p className="text-sm font-semibold text-slate-400">
                    Original X-ray
                  </p>
                  {previewUrl && (
                    <img
                      src={previewUrl}
                      alt="Original X-ray"
                      className="mt-4 aspect-square w-full rounded-2xl bg-black object-contain"
                    />
                  )}
                </div>

                <div className="fade-up rounded-[2rem] border border-white/10 bg-white/[0.03] p-6 shadow-sm">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-semibold text-slate-400">
                      Grad-CAM overlay
                    </p>
                    <Flame className="h-4 w-4 text-sky-300" />
                  </div>
                  <img
                    src={`data:image/png;base64,${result.gradcam.image_base64}`}
                    alt="Grad-CAM"
                    className="mt-4 aspect-square w-full rounded-2xl bg-black object-contain"
                  />
                  <p className="mt-3 text-xs leading-5 text-slate-400">
                    {result.gradcam.explanation}
                  </p>
                </div>
              </div>

              <div className="fade-up rounded-[2rem] border border-white/10 bg-white/[0.03] p-6 shadow-sm">
                <h3 className="text-xl font-semibold">Interpretation</h3>
                <p className="mt-4 leading-7 text-slate-300">
                  {result.interpretation}
                </p>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
