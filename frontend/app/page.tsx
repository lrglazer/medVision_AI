import Link from "next/link";
import {
  Activity,
  ArrowRight,
  Bone,
  FileDown,
  Flame,
  ScanSearch,
  ShieldAlert,
  UploadCloud,
} from "lucide-react";

const specialists = [
  {
    title: "Chest X-ray Analysis",
    subtitle: "14-label thoracic imaging model",
    description:
      "Analyze frontal or lateral chest radiographs across 14 CheXpert observations with explainable AI and structured reporting.",
    href: "/chest",
    icon: Activity,
    metric: "0.8143",
    metricLabel: "Validation macro AUC",
    features: ["14 findings", "DenseNet-121", "Grad-CAM", "PDF report"],
  },
  {
    title: "Bone X-ray Analysis",
    subtitle: "Automatic musculoskeletal routing",
    description:
      "Detect the body region automatically, estimate normal-versus-abnormal appearance, and generate a visual explanation.",
    href: "/bone",
    icon: Bone,
    metric: "0.8706",
    metricLabel: "Abnormality AUC",
    features: ["7 body regions", "96.7% region accuracy", "Grad-CAM", "PDF report"],
  },
];

export default function HomePage() {
  return (
    <div className="overflow-hidden">
      <section className="relative border-b border-slate-200 bg-white">
        <div className="absolute inset-x-0 top-0 -z-0 h-96 bg-[radial-gradient(circle_at_top_left,_rgba(124,58,237,0.12),_transparent_55%)]" />
        <div className="relative mx-auto grid max-w-7xl gap-14 px-4 py-20 sm:px-6 lg:grid-cols-[1.08fr_0.92fr] lg:px-8 lg:py-28">
          <div className="flex flex-col justify-center">
            <div className="inline-flex w-fit items-center gap-2 rounded-full border border-violet-200 bg-violet-50 px-3 py-1.5 text-sm font-semibold text-violet-700">
              <Activity className="h-4 w-4" />
              Medical-imaging AI research platform
            </div>

            <h1 className="mt-7 max-w-4xl text-5xl font-semibold tracking-[-0.04em] text-slate-950 sm:text-6xl lg:text-7xl">
              Explainable AI for medical image analysis
            </h1>

            <p className="mt-7 max-w-2xl text-lg leading-8 text-slate-600">
              MedVision combines deep-learning models, study validation,
              Grad-CAM explanations, and structured reports for chest and
              musculoskeletal radiographs.
            </p>

            <div className="mt-9 flex flex-wrap gap-3">
              <Link
                href="/chest"
                className="inline-flex items-center gap-2 rounded-xl bg-slate-950 px-5 py-3.5 font-semibold text-white shadow-lg shadow-slate-950/10"
              >
                Analyze Chest X-ray
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href="/bone"
                className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-white px-5 py-3.5 font-semibold text-slate-900"
              >
                Analyze Bone X-ray
              </Link>
            </div>

            <div className="mt-10 flex flex-wrap gap-x-8 gap-y-3 text-sm text-slate-500">
              <span>PyTorch</span>
              <span>FastAPI</span>
              <span>Next.js</span>
              <span>Grad-CAM</span>
            </div>
          </div>

          <div className="relative">
            <div className="absolute -inset-8 -z-10 rounded-full bg-violet-200/40 blur-3xl" />
            <div className="rounded-[2rem] border border-slate-800 bg-slate-950 p-7 text-white shadow-2xl">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-semibold uppercase tracking-[0.18em] text-violet-300">
                    Analysis workflow
                  </p>
                  <p className="mt-2 text-sm text-slate-400">
                    From upload to structured result
                  </p>
                </div>
                <div className="h-3 w-3 rounded-full bg-emerald-400 shadow-[0_0_18px_rgba(52,211,153,0.8)]" />
              </div>

              <div className="mt-7 space-y-3">
                {[
                  [UploadCloud, "Upload study", "PNG or JPEG radiograph"],
                  [ScanSearch, "Validate and route", "Confirm supported study type"],
                  [Activity, "Run inference", "Generate structured model outputs"],
                  [Flame, "Explain", "Create Grad-CAM visualization"],
                  [FileDown, "Report", "Review and export a PDF"],
                ].map(([Icon, title, text], index) => {
                  const StepIcon = Icon as typeof Activity;
                  return (
                    <div
                      key={String(title)}
                      className="flex items-center gap-4 rounded-2xl border border-white/10 bg-white/[0.04] p-4"
                    >
                      <div className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-violet-500/20 text-violet-300">
                        <StepIcon className="h-5 w-5" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="font-semibold">
                          {index + 1}. {String(title)}
                        </p>
                        <p className="mt-1 text-sm text-slate-400">
                          {String(text)}
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:px-8">
        <div className="max-w-2xl">
          <p className="text-sm font-semibold uppercase tracking-[0.18em] text-violet-700">
            Supported analysis
          </p>
          <h2 className="mt-3 text-4xl font-semibold tracking-tight text-slate-950">
            Choose an imaging workflow
          </h2>
          <p className="mt-4 leading-7 text-slate-600">
            Each specialist has its own model pipeline, validation metrics,
            explainability view, and report format.
          </p>
        </div>

        <div className="mt-10 grid gap-6 lg:grid-cols-2">
          {specialists.map(
            ({
              title,
              subtitle,
              description,
              href,
              icon: Icon,
              metric,
              metricLabel,
              features,
            }) => (
              <Link
                key={title}
                href={href}
                className="group rounded-[2rem] border border-slate-200 bg-white p-8 shadow-sm transition duration-300 hover:-translate-y-1 hover:border-violet-200 hover:shadow-xl"
              >
                <div className="flex items-start justify-between gap-5">
                  <div className="grid h-14 w-14 place-items-center rounded-2xl bg-violet-100 text-violet-700">
                    <Icon className="h-7 w-7" />
                  </div>
                  <ArrowRight className="h-5 w-5 text-slate-400 transition group-hover:translate-x-1 group-hover:text-violet-700" />
                </div>

                <p className="mt-7 text-sm font-semibold uppercase tracking-[0.14em] text-violet-700">
                  {subtitle}
                </p>
                <h3 className="mt-2 text-3xl font-semibold tracking-tight">
                  {title}
                </h3>
                <p className="mt-4 leading-7 text-slate-600">{description}</p>

                <div className="mt-6 flex flex-wrap gap-2">
                  {features.map((feature) => (
                    <span
                      key={feature}
                      className="rounded-full bg-slate-100 px-3 py-1.5 text-xs font-semibold text-slate-700"
                    >
                      {feature}
                    </span>
                  ))}
                </div>

                <div className="mt-8 border-t border-slate-200 pt-6">
                  <p className="text-4xl font-semibold tracking-tight">{metric}</p>
                  <p className="mt-1 text-sm text-slate-500">{metricLabel}</p>
                </div>
              </Link>
            )
          )}
        </div>

        <div className="mt-12 flex gap-4 rounded-3xl border border-amber-200 bg-amber-50 p-6">
          <ShieldAlert className="mt-0.5 h-6 w-6 shrink-0 text-amber-700" />
          <div>
            <h2 className="font-semibold text-amber-950">
              Educational research use only
            </h2>
            <p className="mt-2 max-w-4xl leading-7 text-amber-900">
              MedVision is not clinical software and must not be used to make
              patient-care decisions. Model outputs may be incorrect and do
              not replace qualified medical interpretation.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
