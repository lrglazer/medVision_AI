"use client";

import { ChangeEvent, DragEvent, useState } from "react";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Download,
  FileImage,
  Flame,
  Loader2,
  RotateCcw,
  ShieldAlert,
  UploadCloud,
} from "lucide-react";

type Finding = {
  name: string;
  score: number;
  display_score: string;
  status:
    | "Positive model finding"
    | "Indeterminate"
    | "Negative model finding";
  display_threshold: string;
};

type ChestResult = {
  study: {
    type: string;
    model: string;
    model_labels: number;
    validation_combined_auc: number;
  };
  findings: Finding[];
  summary_finding: Finding & {
    is_normal_summary: boolean;
    summary_label: string;
    summary_status: Finding["status"];
  };
  findings_text: string;
  impression: string[];
  overall_confidence: string;
  gradcam: {
    finding: string;
    image_base64: string;
    explanation: string;
  };
  disclaimer: string;
};

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export default function Analyzer() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<ChestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

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

      const response = await fetch(`${API_URL}/api/chest/predict`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      const payload = await response.json();
      if (!response.ok) throw new Error(payload?.detail ?? "Analysis failed.");
      setResult(payload as ChestResult);
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

      const response = await fetch(`${API_URL}/api/chest/report`, {
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
      anchor.download = `medvision-${selectedFile.name.replace(/\.[^.]+$/, "")}-chest-report.pdf`;
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
  const summaryFinding = result?.summary_finding ?? null;
  const positive = summaryFinding?.status === "Positive model finding";
  const negative = summaryFinding?.status === "Negative model finding";
  const StatusIcon = negative ? CheckCircle2 : AlertCircle;

  return (
    <section className="mx-auto w-full max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
      <div className="mb-9 max-w-3xl">
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-violet-700">
          Chest X-ray Analysis
        </p>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl">
          Explainable analysis across 14 chest findings
        </h1>
        <p className="mt-4 text-lg leading-8 text-slate-600">
          Upload a frontal or lateral chest radiograph to generate multi-label
          model outputs, Grad-CAM, structured findings, and a PDF report.
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="space-y-5">
          <div className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-sm">
            <div
              onDragEnter={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              className={`flex min-h-72 flex-col items-center justify-center rounded-2xl border-2 border-dashed p-6 text-center transition ${
                isDragging
                  ? "border-violet-500 bg-violet-50"
                  : "border-slate-300 bg-slate-50"
              }`}
            >
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Selected chest X-ray"
                  className="max-h-72 w-full rounded-xl object-contain"
                />
              ) : (
                <>
                  <UploadCloud className="h-9 w-9 text-violet-700" />
                  <p className="mt-4 font-medium text-slate-900">
                    Drag and drop a PNG or JPEG
                  </p>
                  <p className="mt-1 text-sm text-slate-500">Maximum size: 12 MB</p>
                </>
              )}

              <label className="mt-5 inline-flex cursor-pointer items-center gap-2 rounded-xl bg-slate-950 px-4 py-2.5 text-sm font-semibold text-white">
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
              <div className="mt-4 flex items-center justify-between rounded-xl bg-slate-50 px-4 py-3">
                <p className="truncate text-sm font-medium">{selectedFile.name}</p>
                <button type="button" onClick={reset} aria-label="Reset">
                  <RotateCcw className="h-4 w-4 text-slate-500" />
                </button>
              </div>
            )}

            {error && (
              <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800">
                {error}
              </div>
            )}

            <button
              type="button"
              onClick={analyzeImage}
              disabled={!selectedFile || isAnalyzing}
              className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-violet-700 px-5 py-3.5 font-semibold text-white disabled:opacity-50"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Analyzing chest radiograph
                </>
              ) : (
                <>
                  <Activity className="h-5 w-5" />
                  Analyze image
                </>
              )}
            </button>

            {result && (
              <button
                type="button"
                onClick={downloadReport}
                disabled={isDownloading}
                className="mt-3 flex w-full items-center justify-center gap-2 rounded-xl border border-slate-300 bg-white px-5 py-3.5 font-semibold text-slate-900 disabled:opacity-50"
              >
                {isDownloading ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Creating report
                  </>
                ) : (
                  <>
                    <Download className="h-5 w-5" />
                    Download PDF report
                  </>
                )}
              </button>
            )}
          </div>

          <div className="flex gap-3 rounded-2xl border border-amber-200 bg-amber-50 p-4">
            <ShieldAlert className="h-5 w-5 shrink-0 text-amber-700" />
            <p className="text-sm leading-6 text-amber-900">
              Educational research use only. These outputs are not diagnoses or
              calibrated clinical probabilities.
            </p>
          </div>
        </div>

        <div className="space-y-6">
          {!result ? (
            <div className="flex min-h-[520px] items-center justify-center rounded-[2rem] border border-slate-200 bg-white p-8 text-center shadow-sm">
              <div>
                <Activity className="mx-auto h-9 w-9 text-violet-700" />
                <h2 className="mt-5 text-2xl font-semibold">
                  Results will appear here
                </h2>
                <p className="mt-2 text-sm text-slate-500">
                  Upload a supported chest radiograph to begin.
                </p>
              </div>
            </div>
          ) : (
            <>
              <div className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-sm">
                <p className="text-sm font-semibold uppercase tracking-[0.16em] text-violet-700">
                  Study
                </p>
                <div className="mt-4 flex flex-wrap items-end justify-between gap-5">
                  <div>
                    <h2 className="text-3xl font-semibold">{result.study.type}</h2>
                    <p className="mt-2 text-sm text-slate-500">
                      {result.study.model} · {result.study.model_labels} findings
                    </p>
                  </div>
                  <div className="rounded-2xl bg-slate-50 px-4 py-3 text-right">
                    <p className="text-xs text-slate-500">Validation combined AUC</p>
                    <p className="mt-1 text-2xl font-semibold">
                      {result.study.validation_combined_auc.toFixed(4)}
                    </p>
                  </div>
                </div>
              </div>

              {summaryFinding && (
                <div className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-sm">
                  <p className="text-sm text-slate-500">Study summary</p>
                  <div className="mt-3 flex flex-wrap items-start justify-between gap-5">
                    <div>
                      <p className="text-5xl font-semibold tracking-tight">
                        {summaryFinding.name === "No concerning findings detected"
                          ? "✓"
                          : summaryFinding.display_score}
                      </p>
                      <p className="mt-2 text-lg font-semibold">{summaryFinding.name}</p>
                    </div>
                    <div
                      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm font-semibold ${
                        positive
                          ? "border-rose-200 bg-rose-50 text-rose-700"
                          : negative
                          ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                          : "border-amber-200 bg-amber-50 text-amber-700"
                      }`}
                    >
                      <StatusIcon className="h-4 w-4" />
                      {summaryFinding.status}
                    </div>
                  </div>
                </div>
              )}

              <div className="grid gap-6 xl:grid-cols-2">
                <div className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-sm">
                  <p className="text-sm font-semibold text-slate-500">
                    Original X-ray
                  </p>
                  {previewUrl && (
                    <img
                      src={previewUrl}
                      alt="Original chest X-ray"
                      className="mt-4 aspect-square w-full rounded-2xl bg-black object-contain"
                    />
                  )}
                </div>

                <div className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-sm">
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-sm font-semibold text-slate-500">
                      Grad-CAM overlay
                    </p>
                    <Flame className="h-4 w-4 text-violet-700" />
                  </div>
                  <img
                    src={`data:image/png;base64,${result.gradcam.image_base64}`}
                    alt="Grad-CAM"
                    className="mt-4 aspect-square w-full rounded-2xl bg-black object-contain"
                  />
                  <p className="mt-3 text-xs leading-5 text-slate-500">
                    Focused on {result.gradcam.finding}. {result.gradcam.explanation}
                  </p>
                </div>
              </div>

              <div className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-sm">
                <h3 className="text-xl font-semibold">All model outputs</h3>
                <div className="mt-5 space-y-4">
                  {result.findings.map((finding) => (
                    <div key={finding.name}>
                      <div className="flex items-center justify-between gap-4 text-sm">
                        <span className="font-medium">{finding.name}</span>
                        <span>{finding.display_score}</span>
                      </div>
                      <div className="mt-2 h-2 rounded-full bg-slate-100">
                        <div
                          className="h-full rounded-full bg-violet-500"
                          style={{ width: `${Math.max(1, finding.score * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-sm">
                <h3 className="text-xl font-semibold">Findings</h3>
                <p className="mt-4 leading-7 text-slate-700">
                  {result.findings_text}
                </p>

                <h3 className="mt-7 text-xl font-semibold">Impression</h3>
                <div className="mt-4 space-y-3">
                  {result.impression.map((item, index) => (
                    <div
                      key={`${item}-${index}`}
                      className="flex gap-3 rounded-2xl bg-slate-50 p-4"
                    >
                      <span className="font-semibold text-violet-700">
                        {index + 1}
                      </span>
                      <p className="text-sm leading-6 text-slate-700">{item}</p>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}