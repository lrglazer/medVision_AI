import { ChestXray, HandXray, KneeXray, SpineXray } from "./XrayArt";
import Reveal from "./Reveal";

const tiles = [
  { Art: ChestXray, label: "Chest · PA view", finding: "Lung Opacity", score: "72%", tone: "rose", delay: "0s" },
  { Art: HandXray, label: "Wrist · AP view", finding: "Abnormal", score: "81%", tone: "rose", delay: "0.9s" },
  { Art: KneeXray, label: "Knee · AP view", finding: "Region flagged", score: "64%", tone: "amber", delay: "1.8s" },
  { Art: SpineXray, label: "Lumbar spine", finding: "No finding", score: "12%", tone: "emerald", delay: "2.7s" },
];

const toneStyles: Record<string, string> = {
  rose: "border-rose-400/40 bg-rose-500/15 text-rose-200",
  amber: "border-amber-400/40 bg-amber-500/15 text-amber-200",
  emerald: "border-emerald-400/40 bg-emerald-500/15 text-emerald-200",
};

export default function XrayGallery() {
  return (
    <section className="relative mx-auto max-w-7xl px-4 pb-20 sm:px-6 lg:px-8">
      <Reveal>
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-teal-300">Explainability</p>
        <h2 className="mt-3 text-4xl font-semibold tracking-tight text-white">Every result shows its reasoning</h2>
        <p className="mt-4 max-w-2xl leading-7 text-slate-300">
          Grad-CAM heatmaps light up the parts of the X-ray that drove the
          model&apos;s answer. Hover a scan to see it without the overlay.
        </p>
      </Reveal>

      <div className="mt-10 grid grid-cols-2 gap-4 lg:grid-cols-4">
        {tiles.map(({ Art, label, finding, score, tone, delay }, i) => (
          <Reveal key={label} delay={i * 100}>
            <figure className="group relative aspect-[4/5] overflow-hidden rounded-2xl border border-white/10 bg-black transition duration-300 hover:-translate-y-1 hover:border-sky-400/40 hover:shadow-[0_24px_60px_-24px_rgba(56,189,248,0.55)]">
              {/* plain X-ray underneath, heatmap version on top fades out on hover */}
              <Art heat={false} className="absolute inset-0 h-full w-full transition duration-700 group-hover:scale-105" />
              <Art className="absolute inset-0 h-full w-full transition duration-700 group-hover:scale-105 group-hover:opacity-0" />
              <div className="scan-line" style={{ animationDelay: delay }} />
              <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-black/85 via-transparent to-black/30" />
              <span className={`absolute right-3 top-3 rounded-md border px-2 py-1 font-mono text-[10px] font-semibold ${toneStyles[tone]}`}>
                {score}
              </span>
              <figcaption className="absolute inset-x-3 bottom-3">
                <p className="font-mono text-[11px] text-slate-400">{label}</p>
                <p className="mt-0.5 text-sm font-semibold text-white">{finding}</p>
              </figcaption>
            </figure>
          </Reveal>
        ))}
      </div>
      <p className="mt-3 text-xs text-slate-500">Illustrations, not real patient images.</p>
    </section>
  );
}
