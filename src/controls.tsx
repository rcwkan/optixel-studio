import { ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { spline, Wheel } from "./api";

type SliderProps = {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  def: number;
  digits?: number;
  /** Logarithmic track (white balance in Kelvin). */
  log?: boolean;
  track?: string;
  format?: (v: number) => string;
  onChange: (v: number) => void;
};

export function Slider({ label, value, min, max, step = 1, def, digits = 0, log, track, format, onChange }: SliderProps) {
  const toUi = (v: number) => (log ? (Math.log(v / min) / Math.log(max / min)) * 1000 : v);
  const fromUi = (u: number) => (log ? Math.round((min * (max / min) ** (u / 1000)) / 10) * 10 : u);
  return (
    <div className="flex items-center gap-2 py-[3px] text-[11px]" onDoubleClick={() => onChange(def)} title="Double-click to reset">
      <span className="w-[86px] shrink-0 truncate text-neutral-400">{label}</span>
      <input
        type="range"
        className={`ol-range min-w-0 flex-1 ${track ?? ""}`}
        min={log ? 0 : min}
        max={log ? 1000 : max}
        step={log ? 1 : step}
        value={toUi(value)}
        onChange={(e) => onChange(fromUi(+e.target.value))}
      />
      <span className={`w-11 text-right tabular-nums ${value === def ? "text-neutral-500" : "text-neutral-200"}`}>
        {format ? format(value) : `${value > 0 && min < 0 ? "+" : ""}${value.toFixed(digits)}`}
      </span>
    </div>
  );
}

export function Panel({ title, children, open: initial = true, right }: { title: ReactNode; children: ReactNode; open?: boolean; right?: ReactNode }) {
  const [open, setOpen] = useState(initial);
  return (
    <section className="border-b border-neutral-800">
      <header
        className="flex cursor-pointer items-center justify-between gap-2 px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-neutral-300 hover:bg-neutral-800/60 transition-colors"
        onClick={() => setOpen(!open)}
      >
        <span className="flex min-w-0 items-center gap-1.5 truncate">
          <span className="inline-block w-3 shrink-0 text-neutral-500">{open ? "▾" : "▸"}</span>
          {title}
        </span>
        {right != null && (
          <span className="shrink-0 whitespace-nowrap" onClick={(e) => e.stopPropagation()}>
            {right}
          </span>
        )}
      </header>
      {open && <div className="px-3 pb-3">{children}</div>}
    </section>
  );
}

/** Colours that tell sections and functions apart, picked to keep 3:1 against both the light and the dark
 *  backgrounds (icons, swatches, bars); `soft` tints a row behind normal text. Whole class names, for Tailwind. */
export const HUE = {
  blue: { text: "text-blue-500", bg: "bg-blue-500", soft: "bg-blue-500/15", border: "border-blue-500" },
  amber: { text: "text-amber-600", bg: "bg-amber-600", soft: "bg-amber-600/15", border: "border-amber-600" },
  violet: { text: "text-violet-500", bg: "bg-violet-500", soft: "bg-violet-500/15", border: "border-violet-500" },
  emerald: { text: "text-emerald-600", bg: "bg-emerald-600", soft: "bg-emerald-600/15", border: "border-emerald-600" },
  rose: { text: "text-rose-500", bg: "bg-rose-500", soft: "bg-rose-500/15", border: "border-rose-500" },
  cyan: { text: "text-cyan-600", bg: "bg-cyan-600", soft: "bg-cyan-600/15", border: "border-cyan-600" },
  orange: { text: "text-orange-500", bg: "bg-orange-500", soft: "bg-orange-500/15", border: "border-orange-500" },
};
export type Hue = keyof typeof HUE;

/** An outline icon on a 24-unit grid (`d` is its path), drawn in the text colour. */
export function Icon({ d, className = "h-3.5 w-3.5" }: { d: string; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={`shrink-0 ${className}`} fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d={d} />
    </svg>
  );
}

export function Btn({ children, onClick, active, disabled, title, className = "" }: { children: ReactNode; onClick?: () => void; active?: boolean; disabled?: boolean; title?: string; className?: string }) {
  return (
    <button
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={`rounded px-2.5 py-1 text-[11px] font-medium transition-all duration-150 disabled:opacity-40 ${
        active
          ? "bg-[var(--theme-accent,#0ea5e9)] text-[var(--theme-accent-contrast,#ffffff)] shadow-xs font-semibold"
          : "bg-neutral-800 text-neutral-200 hover:bg-neutral-700"
      } ${className}`}
    >
      {children}
    </button>
  );
}

export function Seg<T extends string>({ options, value, onChange }: { options: [T, string][]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="flex overflow-hidden rounded bg-neutral-800 text-[11px]">
      {options.map(([v, label]) => (
        <button key={v} onClick={() => onChange(v)} className={`flex-1 whitespace-nowrap px-1.5 py-1 ${v === value ? "bg-neutral-600 text-white" : "text-neutral-400 hover:text-neutral-200"}`}>
          {label}
        </button>
      ))}
    </div>
  );
}

export function Histogram({ hist }: { hist: Uint32Array[] | null }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !hist) return;
    const g = c.getContext("2d")!;
    const { width: W, height: H } = c;
    g.clearRect(0, 0, W, H);
    let max = 1;
    for (const ch of hist.slice(0, 3)) for (let i = 1; i < 255; i++) max = Math.max(max, ch[i]);
    g.globalCompositeOperation = "lighter";
    ["rgba(230,60,60,.6)", "rgba(60,200,80,.6)", "rgba(60,110,240,.6)"].forEach((col, k) => {
      g.fillStyle = col;
      g.beginPath();
      g.moveTo(0, H);
      for (let i = 0; i < 256; i++) g.lineTo((i / 255) * W, H - Math.min(1, Math.sqrt(hist[k][i] / max)) * H);
      g.lineTo(W, H);
      g.fill();
    });
    g.globalCompositeOperation = "source-over";
  }, [hist]);
  const total = hist ? hist[3].reduce((a, b) => a + b, 0) || 1 : 1;
  const hi = hist ? Math.max(hist[0][255], hist[1][255], hist[2][255]) / total : 0;
  const lo = hist ? hist[3][0] / total : 0;
  return (
    <div>
      <canvas ref={ref} width={512} height={160} className="h-24 w-full rounded bg-neutral-950" />
      <div className="mt-1 flex justify-between text-[10px] tabular-nums text-neutral-500">
        <span className={lo > 0.001 ? "text-sky-400" : ""}>◣ shadows {(lo * 100).toFixed(1)}%</span>
        <span className={hi > 0.001 ? "text-red-400" : ""}>highlights {(hi * 100).toFixed(1)}% ◢</span>
      </div>
    </div>
  );
}

export function ColorWheel({ label, wheel, onChange, size = 92 }: { label: string; wheel: Wheel; onChange: (w: Wheel) => void; size?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const set = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    const x = e.clientX - (r.left + r.width / 2);
    const y = e.clientY - (r.top + r.height / 2);
    onChange({ ...wheel, hue: Math.round(((Math.atan2(y, x) * 180) / Math.PI + 360) % 360), sat: Math.round(Math.min(100, (Math.hypot(x, y) / (r.width / 2)) * 100)) });
  };
  const rad = ((wheel.sat / 100) * size) / 2;
  const a = (wheel.hue * Math.PI) / 180;
  return (
    <div className="flex flex-col items-center gap-1">
      <div
        ref={ref}
        className="relative cursor-crosshair rounded-full"
        style={{
          width: size,
          height: size,
          background: "radial-gradient(circle, rgb(120,120,120) 0%, rgba(120,120,120,0) 72%), conic-gradient(from 90deg, #f33, #ff3, #3f3, #3ff, #33f, #f3f, #f33)",
        }}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          set(e);
        }}
        onPointerMove={(e) => e.buttons & 1 && set(e)}
        onDoubleClick={() => onChange({ ...wheel, hue: 0, sat: 0 })}
      >
        <div className="pointer-events-none absolute h-3 w-3 rounded-full border-2 border-white shadow" style={{ left: size / 2 + Math.cos(a) * rad - 6, top: size / 2 + Math.sin(a) * rad - 6 }} />
      </div>
      <span className="text-[10px] text-neutral-400">
        {label} {wheel.sat > 0 ? `${wheel.hue}° / ${wheel.sat}` : ""}
      </span>
      <input type="range" className="ol-range w-full" min={-100} max={100} value={wheel.lum} onChange={(e) => onChange({ ...wheel, lum: +e.target.value })} onDoubleClick={() => onChange({ ...wheel, lum: 0 })} title="Luminance" />
    </div>
  );
}

const clamp255 = (v: number) => Math.max(0, Math.min(255, Math.round(v)));

/** Point curve editor; points are [in, out] in 0–255. Empty = identity. */
export function CurveEditor({ points, onChange, color = "#ddd", hist }: { points: number[][]; onChange: (p: number[][]) => void; color?: string; hist?: Uint32Array | null }) {
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<number | null>(null);
  const pts = points.length >= 2 ? points : [[0, 0], [255, 255]];
  const path = useMemo(() => {
    const f = spline(pts);
    let d = "";
    for (let x = 0; x <= 255; x += 3) d += `${x ? "L" : "M"}${x} ${255 - clamp255(f(x))}`;
    return d;
  }, [pts]);
  const histPath = useMemo(() => {
    if (!hist) return "";
    let max = 1;
    for (let i = 1; i < 255; i++) max = Math.max(max, hist[i]);
    let d = "M0 255";
    for (let i = 0; i < 256; i++) d += `L${i} ${255 - Math.min(1, Math.sqrt(hist[i] / max)) * 200}`;
    return d + "L255 255Z";
  }, [hist]);
  const pos = (e: React.PointerEvent) => {
    const r = svg.current!.getBoundingClientRect();
    return [clamp255(((e.clientX - r.left) / r.width) * 255), clamp255(255 - ((e.clientY - r.top) / r.height) * 255)];
  };
  return (
    <svg
      ref={svg}
      viewBox="0 0 255 255"
      className="aspect-square w-full cursor-crosshair rounded bg-neutral-950"
      onPointerDown={(e) => {
        const [x, y] = pos(e);
        if (pts.some((p) => Math.abs(p[0] - x) < 8)) return;
        const next = [...pts, [x, y]].sort((a, b) => a[0] - b[0]);
        drag.current = next.findIndex((p) => p[0] === x);
        e.currentTarget.setPointerCapture(e.pointerId);
        onChange(next);
      }}
      onPointerMove={(e) => {
        const i = drag.current;
        if (i === null || !(e.buttons & 1)) return;
        const [x, y] = pos(e);
        const lo = i === 0 ? 0 : pts[i - 1][0] + 1;
        const hi = i === pts.length - 1 ? 255 : pts[i + 1][0] - 1;
        onChange(pts.map((p, k) => (k === i ? [Math.max(lo, Math.min(hi, x)), y] : p)));
      }}
      onPointerUp={() => (drag.current = null)}
    >
      {[64, 128, 192].map((v) => (
        <g key={v} stroke="#2a2a2a" strokeWidth={0.7}>
          <line x1={v} y1={0} x2={v} y2={255} />
          <line x1={0} y1={v} x2={255} y2={v} />
        </g>
      ))}
      {histPath && <path d={histPath} fill="#3a3a3a" />}
      <line x1={0} y1={255} x2={255} y2={0} stroke="#444" strokeWidth={0.7} strokeDasharray="3 3" />
      <path d={path} fill="none" stroke={color} strokeWidth={1.6} />
      {pts.map((p, i) => (
        <circle
          key={i}
          cx={p[0]}
          cy={255 - p[1]}
          r={4}
          fill="#111"
          stroke={color}
          strokeWidth={1.5}
          onPointerDown={(e) => {
            e.stopPropagation();
            drag.current = i;
            svg.current!.setPointerCapture(e.pointerId);
          }}
          onDoubleClick={(e) => {
            e.stopPropagation();
            if (i > 0 && i < pts.length - 1) onChange(pts.filter((_, k) => k !== i));
          }}
        />
      ))}
    </svg>
  );
}

export function Stars({ value, onChange }: { value: number; onChange?: (v: number) => void }) {
  return (
    <span className="tracking-tight">
      {[1, 2, 3, 4, 5].map((s) => (
        <span key={s} onClick={onChange && ((e) => (e.stopPropagation(), onChange(s === value ? 0 : s)))} className={`${onChange ? "cursor-pointer" : ""} ${s <= value ? "text-amber-300" : "text-neutral-600"}`}>
          ★
        </span>
      ))}
    </span>
  );
}
