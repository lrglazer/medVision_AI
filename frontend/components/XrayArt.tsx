/* Stylized, hand-drawn X-ray illustrations (SVG). Not real patient images. */

type ArtProps = { className?: string; heat?: boolean };

function Defs({ id }: { id: string }) {
  return (
    <defs>
      <radialGradient id={`${id}-bg`} cx="50%" cy="45%" r="65%">
        <stop offset="0" stopColor="#1e293b" />
        <stop offset="1" stopColor="#020617" />
      </radialGradient>
      <linearGradient id={`${id}-bone`} x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stopColor="#f8fafc" stopOpacity="0.95" />
        <stop offset="1" stopColor="#94a3b8" stopOpacity="0.75" />
      </linearGradient>
      <radialGradient id={`${id}-heat`} cx="50%" cy="50%" r="50%">
        <stop offset="0" stopColor="#ef4444" stopOpacity="0.9" />
        <stop offset="0.45" stopColor="#f59e0b" stopOpacity="0.6" />
        <stop offset="0.75" stopColor="#22d3ee" stopOpacity="0.25" />
        <stop offset="1" stopColor="#22d3ee" stopOpacity="0" />
      </radialGradient>
      <filter id={`${id}-glow`} x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="2.2" result="b" />
        <feMerge>
          <feMergeNode in="b" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
    </defs>
  );
}

export function HandXray({ className = "", heat = true }: ArtProps) {
  const id = "hand";
  // finger: [x, baseY, angle, lengths]
  const fingers: [number, number, number, number[]][] = [
    [108, 150, -14, [38, 26, 18]],
    [142, 132, -4, [44, 30, 20]],
    [176, 128, 3, [48, 32, 21]],
    [208, 136, 10, [44, 28, 19]],
    [236, 156, 18, [34, 22, 16]],
  ];
  return (
    <svg viewBox="0 0 320 400" className={className} aria-hidden="true">
      <Defs id={id} />
      <rect width="320" height="400" fill={`url(#${id}-bg)`} />
      <ellipse cx="170" cy="230" rx="120" ry="170" fill="#334155" opacity="0.35" />
      <g filter={`url(#${id}-glow)`} fill={`url(#${id}-bone)`}>
        {fingers.map(([x, y, a, lens], i) => {
          let off = 0;
          return (
            <g key={i} transform={`rotate(${a} ${x} ${y + 70})`}>
              <rect x={x - 9} y={y} width="18" height="70" rx="8" />
              {lens.map((l, j) => {
                off += l + 4;
                return <rect key={j} x={x - 7 + j} y={y - off} width={14 - j * 2} height={l} rx="6" />;
              })}
            </g>
          );
        })}
        {/* thumb */}
        <g transform="rotate(-48 92 262)">
          <rect x="82" y="222" width="20" height="52" rx="9" />
          <rect x="84" y="186" width="16" height="32" rx="7" />
          <rect x="86" y="160" width="12" height="22" rx="6" />
        </g>
        {/* carpals */}
        {[[132, 232], [158, 228], [184, 230], [210, 238], [140, 258], [168, 256], [196, 262], [222, 266]].map(([cx, cy], i) => (
          <ellipse key={i} cx={cx} cy={cy} rx="13" ry="11" />
        ))}
        {/* radius & ulna */}
        <rect x="128" y="276" width="36" height="140" rx="14" />
        <rect x="190" y="282" width="26" height="140" rx="12" />
      </g>
      {heat && <ellipse className="heat-pulse" cx="150" cy="300" rx="52" ry="40" fill={`url(#${id}-heat)`} />}
    </svg>
  );
}

export function ChestXray({ className = "", heat = true }: ArtProps) {
  const id = "chest";
  return (
    <svg viewBox="0 0 320 400" className={className} aria-hidden="true">
      <Defs id={id} />
      <rect width="320" height="400" fill={`url(#${id}-bg)`} />
      <ellipse cx="105" cy="210" rx="72" ry="120" fill="#0b1220" opacity="0.9" />
      <ellipse cx="215" cy="210" rx="72" ry="120" fill="#0b1220" opacity="0.9" />
      <ellipse cx="185" cy="245" rx="50" ry="62" fill="#475569" opacity="0.45" />
      <g filter={`url(#${id}-glow)`} fill="none" stroke={`url(#${id}-bone)`} strokeLinecap="round">
        <path d="M160 40 V380" strokeWidth="14" />
        {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
          <g key={i} strokeWidth="5">
            <path d={`M154 ${82 + i * 30} C110 ${66 + i * 30}, 60 ${90 + i * 32}, 40 ${130 + i * 30}`} />
            <path d={`M166 ${82 + i * 30} C210 ${66 + i * 30}, 260 ${90 + i * 32}, 280 ${130 + i * 30}`} />
          </g>
        ))}
        <path d="M120 64 C90 50, 54 52, 26 70" strokeWidth="8" />
        <path d="M200 64 C230 50, 266 52, 294 70" strokeWidth="8" />
      </g>
      {heat && <ellipse className="heat-pulse" cx="95" cy="270" rx="58" ry="46" fill={`url(#${id}-heat)`} />}
    </svg>
  );
}

export function KneeXray({ className = "", heat = true }: ArtProps) {
  const id = "knee";
  return (
    <svg viewBox="0 0 320 400" className={className} aria-hidden="true">
      <Defs id={id} />
      <rect width="320" height="400" fill={`url(#${id}-bg)`} />
      <rect x="70" y="0" width="180" height="400" rx="80" fill="#334155" opacity="0.35" />
      <g filter={`url(#${id}-glow)`} fill={`url(#${id}-bone)`}>
        <path d="M120 0 H200 V150 C230 160, 236 196, 214 206 C196 214, 180 196, 160 198 C140 196, 124 214, 106 206 C84 196, 90 160, 120 150 Z" />
        <ellipse cx="160" cy="172" rx="26" ry="20" opacity="0.7" />
        <path d="M104 224 C124 214, 146 222, 160 220 C174 222, 196 214, 216 224 C220 244, 206 256, 196 262 V400 H126 V262 C114 256, 100 244, 104 224 Z" />
        <rect x="214" y="250" width="16" height="150" rx="8" opacity="0.85" />
      </g>
      {heat && <ellipse className="heat-pulse" cx="190" cy="214" rx="46" ry="30" fill={`url(#${id}-heat)`} />}
    </svg>
  );
}

export function SpineXray({ className = "", heat = true }: ArtProps) {
  const id = "spine";
  return (
    <svg viewBox="0 0 320 400" className={className} aria-hidden="true">
      <Defs id={id} />
      <rect width="320" height="400" fill={`url(#${id}-bg)`} />
      <rect x="60" y="0" width="200" height="400" rx="90" fill="#334155" opacity="0.3" />
      <g filter={`url(#${id}-glow)`} fill={`url(#${id}-bone)`}>
        {Array.from({ length: 9 }).map((_, i) => {
          const y = 14 + i * 43;
          const w = 66 + i * 3;
          return (
            <g key={i}>
              <rect x={160 - w / 2} y={y} width={w} height="32" rx="9" />
              <rect x={160 - w / 2 - 22} y={y + 10} width="22" height="10" rx="5" opacity="0.7" />
              <rect x={160 + w / 2} y={y + 10} width="22" height="10" rx="5" opacity="0.7" />
            </g>
          );
        })}
      </g>
      {heat && <ellipse className="heat-pulse" cx="160" cy="250" rx="54" ry="34" fill={`url(#${id}-heat)`} />}
    </svg>
  );
}
