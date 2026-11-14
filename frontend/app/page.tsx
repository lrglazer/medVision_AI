import Link from "next/link";
import {
  Activity,
  ArrowRight,
  Bone,
  CheckCircle2,
  FileDown,
  Flame,
  ScanSearch,
  ShieldAlert,
  ShieldCheck,
  UploadCloud,
} from "lucide-react";
import ModelStatus from "../components/ModelStatus";
import Reveal from "../components/Reveal";
import CountUp from "../components/CountUp";
import XrayGallery from "../components/XrayGallery";
import { ChestXray, HandXray } from "../components/XrayArt";

const specialists = [
  {
    title: "Chest X-ray",
    subtitle: "Thoracic specialist",
    description:
      "Screens frontal and lateral chest radiographs for 8 common findings, flags normal studies, and explains every result with a heatmap and a PDF report.",
    href: "/chest",
    icon: Activity,
    Art: ChestXray,
    accent: "from-sky-400/30 to-sky-400/0",
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
    Art: HandXray,
    accent: "from-violet-400/30 to-violet-400/0",
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

const tickerItems = [
  "Cardiomegaly", "Lung Opacity", "Pneumonia", "Consolidation", "Edema",
  "Pleural Effusion", "Pneumothorax", "Atelectasis", "Elbow", "Finger",
  "Forearm", "Hand", "Humerus", "Shoulder", "Wrist",
];

const bars = [
  { name: "Lung Opacity", value: "72%", color: "bg-rose-400", width: "w-[72%]", delay: "0.3s" },
  { name: "Pleural Effusion", value: "18%", color: "bg-amber-300", width: "w-[18%]", delay: "0.5s" },
  { name: "Cardiomegaly", value: "9%", color: "bg-emerald-400", width: "w-[9%]", delay: "0.7s" },
];

export default function HomePage() {
  return (
    <div className="relative overflow-hidden">
      {/* drifting color glows */}
      <div className="aurora -left-40 top-10 h-[420px] w-[420px] bg-violet-600/40" />
      <div className="aurora right-[-120px] top-[-80px] h-[480px] w-[480px] bg-sky-500/35 [animation-delay:-6s]" />
      <div className="aurora left-1/3 top-[620px] h-[360px] w-[360px] bg-teal-400/20 [animation-delay:-12s]" />
      <div className="grid-bg pointer-events-none absolute inset-0" />

      <section className="relative mx-auto grid max-w-7xl gap-14 px-4 pb-16 pt-16 sm:px-6 lg:grid-cols-[1.05fr_0.95fr] lg:px-8 lg:pb-24 lg:pt-24">
        <div className="flex flex-col justify-center">
          <div className="fade-up flex flex-wrap items-center gap-3">
            <span className="inline-flex items-center gap-2 rounded-full border border-sky-400/30 bg-sky-400/10 px-3 py-1.5 text-sm font-semibold text-sky-200">
              <Activity className="h-4 w-4" />
              Explainable medical-imaging AI
            </span>
            <ModelStatus />
          </div>

          <h1 className="fade-up mt-7 max-w-4xl text-5xl font-semibold tracking-[-0.04em] text-white [animation-delay:0.1s] sm:text-6xl lg:text-7xl">
            See what the AI <span className="gradient-text-animated">sees</span> in every{" "}
            <span className="whitespace-nowrap">X-ray.</span>
          </h1>

          <p className="fade-up mt-7 max-w-xl text-lg leading-8 text-slate-300 [animation-delay:0.2s]">
            MedVision reads chest and bone radiographs with deep-learning
            specialists, shows its reasoning as a heatmap, and turns the
            result into a structured report in under a minute.
          </p>

          <div className="fade-up mt-9 flex flex-wrap gap-3 [animation-delay:0.3s]">
            <Link
              href="/chest"
              className="group inline-flex items-center gap-2 rounded-xl bg-white px-5 py-3.5 font-semibold text-slate-950 transition hover:-translate-y-0.5 hover:shadow-[0_12px_50px_-10px_rgba(56,189,248,0.8)]"
            >
              Analyze a chest X-ray
              <ArrowRight className="h-4 w-4 transition group-hover:translate-x-1" />
            </Link>
            <Link
              href="/bone"
              className="inline-flex items-center gap-2 rounded-xl border border-white/15 px-5 py-3.5 font-semibold text-white transition hover:-translate-y-0.5 hover:border-white/40 hover:bg-white/5"
            >
              Analyze a bone X-ray
            </Link>
          </div>

          <div className="fade-up mt-10 flex flex-wrap gap-x-8 gap-y-3 text-sm text-slate-400 [animation-delay:0.4s]">
            <span>PyTorch</span>
            <span>DenseNet-121</span>
            <span>FastAPI</span>
            <span>Next.js</span>
            <span>Grad-CAM</span>
          </div>
        </div>

        {/* Animated viewer */}
        <div className="fade-up relative [animation-delay:0.25s]">
          <div className="absolute -inset-10 -z-10 rounded-full bg-sky-500/15 blur-3xl" />

          <div className="float absolute -left-6 top-16 z-20 hidden items-center gap-2 rounded-xl border border-emerald-400/30 bg-[#06120e]/90 px-3 py-2 text-xs font-semibold text-emerald-200 shadow-xl backdrop-blur sm:flex">
            <CheckCircle2 className="h-4 w-4" />
            Chest X-ray verified · 99.9%
          </div>
          <div className="float-delay absolute -right-4 bottom-32 z-20 hidden items-center gap-2 rounded-xl border border-violet-400/30 bg-[#0d0a1a]/90 px-3 py-2 text-xs font-semibold text-violet-200 shadow-xl backdrop-blur sm:flex">
            <Flame className="h-4 w-4" />
            Heatmap ready
          </div>

          <div className="glow-border rounded-[2rem] p-5 shadow-2xl shadow-black/60">
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
                    <stop offset="0" stopColor="#ef4444" stopOpacity="0.9" />
                    <stop offset="0.45" stopColor="#f59e0b" stopOpacity="0.6" />
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
                <ellipse className="heat-pulse" cx="128" cy="190" rx="72" ry="54" fill="url(#heat)" />
                <g className="bracket" stroke="#38bdf8" strokeWidth="2.5" fill="none" strokeLinecap="round">
                  <path d="M54 146 V134 H66" />
                  <path d="M190 134 H202 V146" />
                  <path d="M202 234 V246 H190" />
                  <path d="M66 246 H54 V234" />
                </g>
                <g fill="#38bdf8" opacity="0.8">
                  {[[300, 90], [330, 150], [280, 210], [250, 120]].map(([x, y]) => (
                    <circle key={`${x}-${y}`} cx={x} cy={y} r="1.6" className="bracket" />
                  ))}
                </g>
              </svg>
              <div className="scan-line" />
              <div className="absolute right-3 top-3 rounded-md border border-sky-400/40 bg-sky-400/10 px-2 py-1 font-mono text-[10px] font-semibold tracking-wider text-sky-200">
                ANALYZING
              </div>
              <div className="absolute bottom-3 left-3 rounded-lg border border-white/10 bg-black/60 px-2.5 py-1.5 font-mono text-[11px] text-slate-300 backdrop-blur">
                Grad-CAM · Lung Opacity
              </div>
            </div>

            <div className="mt-4 grid grid-cols-3 gap-3">
              {bars.map(({ name, value, color, width, delay }) => (
                <div key={name} className="rounded-xl border border-white/10 bg-white/[0.03] p-3">
                  <p className="truncate text-xs text-slate-400">{name}</p>
                  <p className="mt-1 font-mono text-lg font-semibold text-white">{value}</p>
                  <div className="mt-2 h-1 overflow-hidden rounded-full bg-white/10">
                    <div className={`bar-grow h-full rounded-full ${color} ${width}`} style={{ animationDelay: delay }} />
                  </div>
                </div>
              ))}
            </div>
            <p className="mt-3 px-1 text-[11px] text-slate-500">Illustration of the results view.</p>
          </div>
        </div>
      </section>

      {/* ticker */}
      <div className="relative overflow-hidden border-y border-white/10 bg-white/[0.02] py-4" aria-hidden="true">
        <div className="ticker flex w-max gap-10 whitespace-nowrap font-mono text-sm text-slate-500">
          {[...tickerItems, ...tickerItems].map((item, i) => (
            <span key={`${item}-${i}`} className="flex items-center gap-10">
              <span className={i % 3 === 0 ? "text-sky-300" : ""}>{item}</span>
              <span className="text-slate-700">◆</span>
            </span>
          ))}
        </div>
      </div>

      {/* stats */}
      <section className="relative mx-auto grid max-w-7xl grid-cols-2 gap-4 px-4 py-16 sm:px-6 lg:grid-cols-4 lg:px-8">
        {[
          { value: 0.882, decimals: 3, suffix: "", label: "Chest validation AUC" },
          { value: 96.7, decimals: 1, suffix: "%", label: "Bone region accuracy" },
          { value: 8, decimals: 0, suffix: "", label: "Chest findings" },
          { value: 7, decimals: 0, suffix: "", label: "Bone regions" },
        ].map((s, i) => (
          <Reveal key={s.label} delay={i * 90}>
            <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-6">
              <p className="font-mono text-4xl font-semibold tracking-tight text-white">
                <CountUp value={s.value} decimals={s.decimals} suffix={s.suffix} />
              </p>
              <p className="mt-2 text-sm text-slate-400">{s.label}</p>
            </div>
          </Reveal>
        ))}
      </section>

      <XrayGallery />

      <section className="relative mx-auto max-w-7xl px-4 pb-20 sm:px-6 lg:px-8">
        <Reveal>
          <p className="text-sm font-semibold uppercase tracking-[0.18em] text-sky-300">Specialists</p>
          <h2 className="mt-3 text-4xl font-semibold tracking-tight text-white">Pick an imaging workflow</h2>
        </Reveal>

        <div className="mt-10 grid gap-6 lg:grid-cols-2">
          {specialists.map(({ title, subtitle, description, href, icon: Icon, Art, accent, iconStyle, metric, metricLabel, features }, i) => (
            <Reveal key={title} delay={i * 120}>
              <Link
                href={href}
                className="group relative block overflow-hidden rounded-[2rem] border border-white/10 bg-white/[0.03] p-8 transition duration-300 hover:-translate-y-1.5 hover:border-white/25 hover:shadow-[0_30px_80px_-30px_rgba(56,189,248,0.45)]"
              >
                <div className="pointer-events-none absolute -right-10 top-6 h-56 w-44 overflow-hidden rounded-2xl opacity-25 transition duration-500 [mask-image:linear-gradient(to_left,black,transparent)] group-hover:opacity-60 sm:h-64 sm:w-52">
                  <Art className="h-full w-full" />
                  <div className="scan-line" />
                </div>
                <div className={`pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-gradient-to-b ${accent} blur-2xl transition duration-500 group-hover:scale-125`} />
                <div className="relative flex items-start justify-between gap-5">
                  <div className={`grid h-14 w-14 place-items-center rounded-2xl ${iconStyle} transition duration-300 group-hover:rotate-6 group-hover:scale-110`}>
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
            </Reveal>
          ))}
        </div>
      </section>

      <section className="relative mx-auto max-w-7xl px-4 pb-24 sm:px-6 lg:px-8">
        <Reveal>
          <p className="text-sm font-semibold uppercase tracking-[0.18em] text-violet-300">How it works</p>
          <h2 className="mt-3 text-4xl font-semibold tracking-tight text-white">From upload to report</h2>
        </Reveal>

        <div className="relative mt-10">
          <div className="absolute left-0 right-0 top-[38px] hidden h-px bg-gradient-to-r from-sky-400/0 via-sky-400/50 to-violet-400/0 lg:block" />
          <div className="relative grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
            {steps.map(({ icon: Icon, title, text }, index) => (
              <Reveal key={title} delay={index * 110}>
                <div className="group h-full rounded-2xl border border-white/10 bg-[#080b14] p-5 transition hover:-translate-y-1 hover:border-sky-400/40">
                  <div className="flex items-center justify-between">
                    <div className="grid h-10 w-10 place-items-center rounded-xl bg-white/[0.06] text-sky-300 transition group-hover:bg-sky-400/20 group-hover:shadow-[0_0_24px_rgba(56,189,248,0.5)]">
                      <Icon className="h-5 w-5" />
                    </div>
                    <span className="font-mono text-xs text-slate-500">0{index + 1}</span>
                  </div>
                  <p className="mt-5 font-semibold text-white">{title}</p>
                  <p className="mt-1.5 text-sm leading-6 text-slate-400">{text}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>

        <Reveal>
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
        </Reveal>
      </section>
    </div>
  );
}
