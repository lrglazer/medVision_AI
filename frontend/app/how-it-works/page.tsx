import {
  FileDown,
  Flame,
  ScanSearch,
  ShieldCheck,
  UploadCloud,
} from "lucide-react";

const steps = [
  {
    icon: UploadCloud,
    title: "Upload a radiograph",
    text: "Choose a supported PNG or JPEG chest or musculoskeletal X-ray.",
  },
  {
    icon: ShieldCheck,
    title: "Validate the study type",
    text: "The input is checked before it reaches the specialist analysis model.",
  },
  {
    icon: ScanSearch,
    title: "Run model inference",
    text: "The selected deep-learning model generates structured image-level outputs.",
  },
  {
    icon: Flame,
    title: "Generate Grad-CAM",
    text: "A heatmap highlights regions that influenced the model output. It is not lesion segmentation.",
  },
  {
    icon: FileDown,
    title: "Review the report",
    text: "Results, metrics, limitations, and downloadable reports are presented in one workflow.",
  },
];

export default function HowItWorksPage() {
  return (
    <section className="mx-auto max-w-5xl px-4 py-16 sm:px-6 lg:px-8">
      <p className="text-sm font-semibold uppercase tracking-[0.18em] text-violet-700">
        How It Works
      </p>
      <h1 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl">
        From uploaded X-ray to explainable model output
      </h1>
      <p className="mt-6 max-w-3xl text-lg leading-8 text-slate-600">
        MedVision combines study validation, specialist models,
        validation-derived thresholds, Grad-CAM, and structured reporting.
      </p>

      <div className="mt-10 space-y-5">
        {steps.map(({ icon: Icon, title, text }, index) => (
          <div
            key={title}
            className="flex gap-5 rounded-[2rem] border border-slate-200 bg-white p-7 shadow-sm"
          >
            <div className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl bg-violet-100 text-violet-700">
              <Icon className="h-6 w-6" />
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-violet-700">
                Step {index + 1}
              </p>
              <h2 className="mt-1 text-xl font-semibold">{title}</h2>
              <p className="mt-2 leading-7 text-slate-600">{text}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-10 rounded-[2rem] border border-amber-200 bg-amber-50 p-6">
        <h2 className="font-semibold text-amber-950">Important limitations</h2>
        <p className="mt-2 leading-7 text-amber-900">
          Model outputs may not generalize across scanners, institutions,
          patient populations, acquisition settings, or unsupported study
          types. MedVision is an educational research project, not clinical
          software.
        </p>
      </div>
    </section>
  );
}
