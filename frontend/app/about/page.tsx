import {
  Activity,
  BrainCircuit,
  Code2,
  Database,
  ShieldAlert,
} from "lucide-react";

const technologies = [
  ["PyTorch", "Deep-learning model development and inference"],
  ["FastAPI", "Python API and model-serving backend"],
  ["Next.js", "Responsive web application and analysis interface"],
  ["Grad-CAM", "Visual explanation of model attention"],
];

export default function AboutPage() {
  return (
    <section className="mx-auto max-w-6xl px-4 py-16 sm:px-6 lg:px-8">
      <p className="text-sm font-semibold uppercase tracking-[0.18em] text-violet-700">
        About MedVision
      </p>
      <h1 className="mt-3 max-w-4xl text-4xl font-semibold tracking-tight sm:text-5xl">
        An explainable AI platform for medical-image research
      </h1>
      <p className="mt-6 max-w-3xl text-lg leading-8 text-slate-600">
        MedVision demonstrates an end-to-end medical-imaging workflow:
        dataset preparation, model training, validation, API development,
        explainability, structured reporting, and frontend design.
      </p>

      <div className="mt-10 grid gap-6 md:grid-cols-3">
        {[
          [BrainCircuit, "Deep learning", "DenseNet-121 models support chest multi-label analysis, musculoskeletal body-part detection, and abnormality estimation."],
          [Activity, "Explainability", "Grad-CAM heatmaps show which image regions influenced model output without claiming lesion segmentation."],
          [ShieldAlert, "Responsible presentation", "The platform separates model output from diagnosis and presents limitations throughout the experience."],
        ].map(([Icon, title, text]) => {
          const CardIcon = Icon as typeof Activity;
          return (
            <div
              key={String(title)}
              className="rounded-[2rem] border border-slate-200 bg-white p-7 shadow-sm"
            >
              <CardIcon className="h-6 w-6 text-violet-700" />
              <h2 className="mt-5 text-xl font-semibold">{String(title)}</h2>
              <p className="mt-3 text-sm leading-7 text-slate-600">
                {String(text)}
              </p>
            </div>
          );
        })}
      </div>

      <div className="mt-12 rounded-[2rem] border border-slate-200 bg-white p-8 shadow-sm">
        <div className="flex items-center gap-3">
          <Code2 className="h-6 w-6 text-violet-700" />
          <h2 className="text-2xl font-semibold">Technology stack</h2>
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-2">
          {technologies.map(([name, description]) => (
            <div
              key={name}
              className="rounded-2xl border border-slate-200 bg-slate-50 p-5"
            >
              <p className="font-semibold">{name}</p>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                {description}
              </p>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-8 flex gap-4 rounded-[2rem] border border-violet-200 bg-violet-50 p-6">
        <Database className="mt-0.5 h-6 w-6 shrink-0 text-violet-700" />
        <p className="leading-7 text-violet-950">
          Chest modeling uses CheXpert. Musculoskeletal modeling uses MURA.
          Displayed performance reflects this project&apos;s validation results
          and should not be interpreted as clinical performance.
        </p>
      </div>
    </section>
  );
}
