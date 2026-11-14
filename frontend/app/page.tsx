import Link from "next/link";
import {
  Activity,
  ArrowRight,
  Bone,
  FileDown,
  Flame,
  ScanSearch,
  ShieldAlert,
  ShieldCheck,
  UploadCloud,
} from "lucide-react";
import ModelStatus from "../components/ModelStatus";

const specialists = [
  {
    title: "Chest X-ray",
    subtitle: "Thoracic specialist",
    description:
      "Screens frontal and lateral chest radiographs for 8 common findings, flags normal studies, and explains every result with a heatmap and a PDF report.",
    href: "/chest",
    icon: Activity,
    accent: "from-sky-400/25 to-sky-400/0",
    iconStyle: "bg-sky-400/15 text-sky-300",
    metric: "0.882",
    metricLabel: "Validation AUC",
    features: ["8 findings", "Normal gate", "Grad-CAM", "PDF report"],
  },
  {
    title: "Bone X-ray",
    subtitle: "Musculoskeletal specialist",
    description:
      "Identifies which of 7 upper-limb regions is in the image, estimates normal vs. abnormal appearance, and highlights the regions that drove the result.",
    href: "/bone",
    icon: Bone,
    accent: "from-violet-400/25 to-violet-400/0",
    iconStyle: "bg-violet-400/15 text-violet-300",
    metric: "96.7%",
    metricLabel: "Body-region accuracy",
    features: ["7 regions", "0.871 AUC", "Grad-CAM", "Auto routing"],
  },
];

const steps = [
  { icon: UploadCloud, title: "Upload", text: "Drop in a PNG or JPEG radiograph." },
  { icon: ShieldCheck, title: "Validate", text: "A gatekeeper model rejects photos, screenshots and the wrong study type." },
  { icon: ScanSearch, title: "Analyze", text: "A DenseNet-121 specialist scores every finding." },
  { icon: Flame, title: "Explain", text: "Grad-CAM shows where the model was looking." },
  { icon: FileDown, title: "Report", text: "Export a structured PDF report." },
];

export default function HomePage() {
  return (
    <div className="relative overflow-hidden">
      <div className="grid-bg pointer-events-none absolute inset-0" />

      <section className="relative mx-auto grid max-w-7xl gap-14 px-4 pb-20 pt-16 sm:px-6 lg:grid-cols-[1.05fr_0.95fr] lg:px-8 lg:pb-28 lg:pt-24">
        <div className="flex flex-col justify-center">
          <div className="flex flex-wrap items-center gap-3">
            <span className="inline-flex items-center gap-2 rounded-full border border-sky-400/30 bg-sky-400/10 px-3 py-1.5 text-sm font-semibold text-sky-200">
              <Activity className="h-4 w-4" />
              Explainable medical-imaging AI
            </span>
            <ModelStatus />
          </div>

          <h1 className="mt-7 max-w-4xl text-5xl font-semibold tracking-[-0.04em] text-white sm:text-6xl lg:text-7xl">
            See what the AI <span className="gradient-text">sees</span> in every X-ray.
          </h1>

          <p className="mt-7 max-w-xl text-lg leading-8 text-slate-300">
            MedVision reads chest and bone radiographs with deep-learning
            specialists, shows its reasoning as a heatmap, and turns the
            result into a structured report in under a minute.
          </p>

          <div className="mt-9 flex flex-wrap gap-3">
            <Link
              href="/chest"
              className="inline-flex items-center gap-2 rounded-xl bg-white px-5 py-3.5 font-semibold text-slate-950 transition hover:-translate-y-0.5 hover:shadow-[0_10px_40px_-10px_rgba(56,189,248,0.6)]"
            >
              Analyze a chest X-ray
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/bone"
              className="inline-flex items-center gap-2 rounded-xl border border-white/15 px-5 py-3.5 font-semibold text-white transition hover:border-white/40"
            >
              Analyze a bone X-ray
            </Link>
          </div>

          <div className="mt-10 flex flex-wrap gap-x-8 gap-y-3 text-sm text-slate-400">
            <span>PyTorch</span>
            <span>DenseNet-121</span>
            <span>FastAPI</span>
            <span>Next.js</span>
            <span>Grad-CAM</span>
          </div>
        </div>

        {/* Viewer mockup */}
        <div className="relative">
          <div className="absolute -inset-10 -z-10 rounded-full bg-sky-500/10 blur-3xl" />
          <div className="rounded-[2rem] border border-white/10 bg-[#0a0e18]/90 p-5 shadow-2xl shadow-black/60">
            <div className="flex items-center justify-between px-1 pb-4">
              <div className="flex gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-rose-400/80" />
                <span className="h-2.5 w-2.5 rounded-full bg-amber-300/80" />
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-400/80" />
              </div>
              <span className="font-mono text-xs text-slate-500">study_0421 · PA chest</span>
            </div>

            <div className="relative aspect-[4/3] overflow-hidden rounded-2xl border border-white/10 bg-black">
              <svg viewBox="0 0 400 300" className="absolute inset-0 h-full w-full" aria-hidden="true">
                <defs>
                  <radialGradient id="lung" cx="50%" cy="45%" r="60%">
                    <stop offset="0" stopColor="#1f2937" />
                    <stop offset="1" stopColor="#030712" />
                  </radialGradient>
                  <radialGradient id="heat" cx="50%" cy="50%" r="50%">
                    <stop offset="0" stopColor="#ef4444" stopOpacity="0.85" />
                    <stop offset="0.45" stopColor="#f59e0b" stopOpacity="0.55" />
                    <stop offset="0.75" stopColor="#22d3ee" stopOpacity="0.25" />
                    <stop offset="1" stopColor="#22d3ee" stopOpacity="0" />
                  </radialGradient>
                </defs>
                <rect width="400" height="300" fill="url(#lung)" />
                <g fill="none" stroke="#cbd5e1" strokeOpacity="0.55" strokeWidth="2.5">
                  <path d="M200 30 V270" strokeOpacity="0.7" strokeWidth="6" />
                  {[0, 1, 2, 3, 4, 5, 6].map((i) => (
                    <g key={i}>
                      <path d={`M196 ${62 + i * 26} C150 ${48 + i * 26}, 104 ${66 + i * 28}, 88 ${100 + i * 26}`} />
                      <path d={`M204 ${62 + i * 26} C250 ${48 + i * 26}, 296 ${66 + i * 28}, 312 ${100 + i * 26}`} />
                    </g>
                  ))}
                  <path d="M150 52 C120 40, 92 44, 70 58" strokeWidth="4" />
                  <path d="M250 52 C280 40, 308 44, 330 58" strokeWidth="4" />
                </g>
                <ellipse cx="128" cy="190" rx="70" ry="52" fill="url(#heat)" />
              </svg>
              <div className="scan-line" />
              <div className="absolute bottom-3 left-3 rounded-lg border border-white/10 bg-black/60 px-2.5 py-1.5 font-mono text-[11px] text-slate-300 backdrop-blur">
                Grad-CAM · Lung Opacity
              </div>
            </div>

            <div className="mt-4 grid grid-cols-3 gap-3">
              {[
                ["Lung Opacity", "72%", "bg-rose-400", "w-[72%]"],
                ["Pleural Effusion", "18%", "bg-amber-300", "w-[18%]"],
                ["Cardiomegaly", "9%", "bg-emerald-400", "w-[9%]"],
              ].map(([name, value, color, width]) => (
                <div key={name} className="rounded-xl border border-white/10 bg-white/[0.03] p-3">
                  <p className="truncate text-xs text-slate-400">{name}</p>
                  <p className="mt-1 font-mono text-lg font-semibold text-white">{value}</p>
                  <div className="mt-2 h-1 rounded-full bg-white/10">
                    <div className={`h-full rounded-full ${color} ${width}`} />
                  </div>
                </div>
              ))}
            </div>
            <p className="mt-3 px-1 text-[11px] text-slate-500">Illustration of the results view.</p>
          </div>
        </div>
      </section>

      <section className="relative mx-auto max-w-7xl px-4 pb-20 sm:px-6 lg:px-8">
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-sky-300">Specialists</p>
        <h2 className="mt-3 text-4xl font-semibold tracking-tight text-white">Pick an imaging workflow</h2>

        <div className="mt-10 grid gap-6 lg:grid-cols-2">
          {specialists.map(({ title, subtitle, description, href, icon: Icon, accent, iconStyle, metric, metricLabel, features }) => (
            <Link
              key={title}
              href={href}
              className="group relative overflow-hidden rounded-[2rem] border border-white/10 bg-white/[0.03] p-8 transition duration-300 hover:-translate-y-1 hover:border-white/25"
            >
              <div className={`pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-gradient-to-b ${accent} blur-2xl`} />
              <div className="relative flex items-start justify-between gap-5">
                <div className={`grid h-14 w-14 place-items-center rounded-2xl ${iconStyle}`}>
                  <Icon className="h-7 w-7" />
                </div>
                <ArrowRight className="h-5 w-5 text-slate-500 transition group-hover:translate-x-1 group-hover:text-white" />
              </div>

              <p className="relative mt-7 text-sm font-semibold uppercase tracking-[0.14em] text-slate-400">{subtitle}</p>
              <h3 className="relative mt-2 text-3xl font-semibold tracking-tight text-white">{title}</h3>
              <p className="relative mt-4 leading-7 text-slate-300">{description}</p>

              <div className="relative mt-6 flex flex-wrap gap-2">
                {features.map((feature) => (
                  <span key={feature} className="rounded-full border border-white/10 bg-white/[0.05] px-3 py-1.5 text-xs font-semibold text-slate-200">
                    {feature}
                  </span>
                ))}
              </div>

              <div className="relative mt-8 border-t border-white/10 pt-6">
                <p className="font-mono text-4xl font-semibold tracking-tight text-white">{metric}</p>
                <p className="mt-1 text-sm text-slate-400">{metricLabel}</p>
              </div>
            </Link>
          ))}
        </div>
      </section>

      <section className="relative mx-auto max-w-7xl px-4 pb-20 sm:px-6 lg:px-8">
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-violet-300">How it works</p>
        <h2 className="mt-3 text-4xl font-semibold tracking-tight text-white">From upload to report</h2>
        <ol className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          {steps.map(({ icon: Icon, title, text }, index) => (
            <li key={title} className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
              <div className="flex items-center justify-between">
                <div className="grid h-10 w-10 place-items-center rounded-xl bg-white/[0.06] text-sky-300">
                  <Icon className="h-5 w-5" />
                </div>
                <span className="font-mono text-xs text-slate-500">0{index + 1}</span>
              </div>
              <p className="mt-5 font-semibold text-white">{title}</p>
              <p className="mt-1.5 text-sm leading-6 text-slate-400">{text}</p>
            </li>
          ))}
        </ol>

        <div className="mt-12 flex gap-4 rounded-3xl border border-amber-400/30 bg-amber-500/10 p-6">
          <ShieldAlert className="mt-0.5 h-6 w-6 shrink-0 text-amber-300" />
          <div>
            <h2 className="font-semibold text-amber-100">Educational research use only</h2>
            <p className="mt-2 max-w-4xl leading-7 text-amber-100/80">
              MedVision is not clinical software and must not be used to make
              patient-care decisions. Model outputs may be incorrect and do not
              replace qualified medical interpretation.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
