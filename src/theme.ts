import { Theme } from "./api";
export type { Theme };

export interface ThemeInfo {
  id: Theme;
  label: string;
  tagline: string;
  category: "Auto" | "Neutral" | "Vibrant" | "Modern" | "Warm" | "Dark" | "Light";
  mode: "dark" | "light" | "system";
  description: string;
  icon: string;
  colors: {
    bg: string;
    card: string;
    border: string;
    text: string;
    accent: string;
    accentGlow: string;
    secondaryAccent?: string;
  };
}

export const THEMES: ThemeInfo[] = [
  {
    id: "system",
    label: "System",
    tagline: "Dynamic OS Match",
    category: "Auto",
    mode: "system",
    description: "Follows your operating system's light or dark appearance preference in real time.",
    icon: "💻",
    colors: {
      bg: "#161616",
      card: "#222222",
      border: "#3a3a3a",
      text: "#d4d4d4",
      accent: "#38bdf8",
      accentGlow: "rgba(56, 189, 248, 0.3)",
      secondaryAccent: "#818cf8",
    },
  },
  {
    id: "dark",
    label: "Studio Dark",
    tagline: "Colorist Standard",
    category: "Neutral",
    mode: "dark",
    description: "Precision neutral dark charcoal (#161616) — zero chromatic bias for critical grading.",
    icon: "🌙",
    colors: {
      bg: "#161616",
      card: "#242424",
      border: "#3d3d3d",
      text: "#d4d4d4",
      accent: "#0ea5e9",
      accentGlow: "rgba(14, 165, 233, 0.35)",
      secondaryAccent: "#38bdf8",
    },
  },
  {
    id: "light",
    label: "Studio Light",
    tagline: "High-Key Clean",
    category: "Neutral",
    mode: "light",
    description: "Crisp bright surfaces with sapphire blue accents for daytime editing & print matching.",
    icon: "☀️",
    colors: {
      bg: "#f4f5f8",
      card: "#ffffff",
      border: "#e2e8f0",
      text: "#0f172a",
      accent: "#2563eb",
      accentGlow: "rgba(37, 99, 235, 0.2)",
      secondaryAccent: "#0284c7",
    },
  },
  {
    id: "slate",
    label: "Cool Slate",
    tagline: "Oceanic Navy",
    category: "Modern",
    mode: "dark",
    description: "Deep oceanic blue-slate tones with bright cyan highlights for a sleek pro studio vibe.",
    icon: "🔷",
    colors: {
      bg: "#0b1329",
      card: "#111c38",
      border: "#1e293b",
      text: "#e2e8f0",
      accent: "#38bdf8",
      accentGlow: "rgba(56, 189, 248, 0.35)",
      secondaryAccent: "#06b6d4",
    },
  },
  {
    id: "warm",
    label: "Warm Darkroom",
    tagline: "Analog Amber & Film",
    category: "Warm",
    mode: "dark",
    description: "Rich dark espresso and mahogany tones with glowing amber gold; gentle on the eyes.",
    icon: "🕯️",
    colors: {
      bg: "#181412",
      card: "#231c19",
      border: "#3a2e28",
      text: "#e7e5e4",
      accent: "#f59e0b",
      accentGlow: "rgba(245, 158, 11, 0.35)",
      secondaryAccent: "#fbbf24",
    },
  },
  {
    id: "nordic",
    label: "Nordic Frost",
    tagline: "Arctic Petrol & Mint",
    category: "Vibrant",
    mode: "dark",
    description: "Petrol midnight navy with refreshing arctic ice-cyan accents — razor-sharp & modern.",
    icon: "❄️",
    colors: {
      bg: "#07131b",
      card: "#0d202d",
      border: "#18374a",
      text: "#e0f2fe",
      accent: "#06b6d4",
      accentGlow: "rgba(6, 182, 212, 0.35)",
      secondaryAccent: "#22d3ee",
    },
  },
  {
    id: "slate_light",
    label: "Alabaster Slate",
    tagline: "Porcelain & Cool Slate",
    category: "Modern",
    mode: "light",
    description: "Crisp cool porcelain and slate surfaces with vibrant oceanic azure highlights.",
    icon: "🏛️",
    colors: {
      bg: "#f1f5f9",
      card: "#ffffff",
      border: "#cbd5e1",
      text: "#0f172a",
      accent: "#0284c7",
      accentGlow: "rgba(2, 132, 199, 0.25)",
      secondaryAccent: "#38bdf8",
    },
  },
  {
    id: "warm_light",
    label: "Warm Linen",
    tagline: "Natural Parchment & Amber",
    category: "Warm",
    mode: "light",
    description: "Natural warm linen and parchment surfaces with analog golden amber accents.",
    icon: "📜",
    colors: {
      bg: "#faf7f2",
      card: "#ffffff",
      border: "#ebe6df",
      text: "#1c1917",
      accent: "#d97706",
      accentGlow: "rgba(217, 119, 6, 0.25)",
      secondaryAccent: "#f59e0b",
    },
  },
  {
    id: "nordic_light",
    label: "Nordic Snow",
    tagline: "Glacier Ice & Mint",
    category: "Vibrant",
    mode: "light",
    description: "Crisp glacial snow surfaces with invigorating arctic ice & seafoam cyan accents.",
    icon: "🏔️",
    colors: {
      bg: "#f0fdfa",
      card: "#ffffff",
      border: "#ccfbf1",
      text: "#082f49",
      accent: "#0891b2",
      accentGlow: "rgba(8, 145, 178, 0.25)",
      secondaryAccent: "#06b6d4",
    },
  },
];

export function getStoredTheme(): Theme {
  try {
    const t = localStorage.getItem("optixel-theme") as Theme | null;
    if (
      t &&
      [
        "system",
        "dark",
        "light",
        "slate",
        "warm",
        "nordic",
        "cyber",
        "emerald",
        "oled",
        "slate_light",
        "warm_light",
        "nordic_light",
      ].includes(t)
    ) {
      return t;
    }
  } catch (_) {}
  return "system";
}

export function isSystemDark(): boolean {
  if (typeof window === "undefined") return true;
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

export function isLightTheme(theme: Theme): boolean {
  if (theme === "system") return !isSystemDark();
  return (
    theme === "light" ||
    theme === "slate_light" ||
    theme === "warm_light" ||
    theme === "nordic_light"
  );
}

export function applyTheme(theme: Theme) {
  try {
    localStorage.setItem("optixel-theme", theme);
  } catch (_) {}

  const root = document.documentElement;
  root.setAttribute("data-theme", theme);

  const isLight = isLightTheme(theme);

  let meta = document.querySelector('meta[name="color-scheme"]') as HTMLMetaElement | null;
  if (!meta) {
    meta = document.createElement("meta");
    meta.name = "color-scheme";
    document.head.appendChild(meta);
  }
  meta.content = theme === "system" ? "light dark" : isLight ? "light" : "dark";
}
