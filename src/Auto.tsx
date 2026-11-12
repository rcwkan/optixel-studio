import { ReactNode, useState } from "react";
import { Btn, Panel, Slider } from "./controls";

/** The 15 professional photographic color grading styles (from 15_photo_styles_color_grading_engine.rs) */
export const PROFILES: [string, string, string][] = [
  // Group A: 10 Professional Color Grading Styles
  ["bleach_bypass", "Bleach Bypass", "Silver-retention simulation: high contrast, desaturated cold slate shadows"],
  ["cyberpunk", "Cyberpunk", "Dual-tone neon grade: electric cyan and magenta over deep violet shadows"],
  ["moody_matte", "Moody Matte", "Lifted matte blacks, olive-drab desaturated greens, and warm amber skin"],
  ["golden_hour", "Golden Hour", "Chocolate-brown shadows, warm golden midtones, creamy peach highlights"],
  ["nordic_noir", "Nordic Noir", "Prussian-blue shadows, crisp steel-gray midtones, and muted warm hues"],
  ["kodak_portra", "Portra 400", "Analog C-41 response: soft contrast, pastel cyan skies, emerald greens, golden skin"],
  ["fuji_velvia", "Fuji Velvia", "Reversal slide film: deep ink blacks, steep contrast, punchy greens and rich reds"],
  ["emerald_gold", "Emerald & Gold", "Matrix/Fincher aesthetic: forest-green shadows, olive midtones, warm gold"],
  ["pastel_airy", "Pastel Airy", "Airy high-key exposure: lifted milky shadows, negative contrast, pastel tones"],
  ["cross_processed", "Cross-Process", "C-41 slide cross-processing: asymmetric curves, cyan shadows, yellow highlights"],

  // Group B: Classic Teal & Orange + 5 Variations
  ["teal_orange", "Teal & Orange", "Classic cinematic: teal shadows, amber highlights, deep shadows, held-back highlights"],
  ["teal_orange_muted", "Matte T&O", "Dusty slate-blue shadows, soft terracotta highlights, and lifted matte blacks"],
  ["teal_orange_dark", "Dark T&O", "Abyssal cyan shadows, fiery bronze-amber highlights, and deep crushed blacks"],
  ["teal_orange_airy", "Bright T&O", "High-luminance mint/aqua shadows with soft pastel coral/peach highlights"],
  ["teal_orange_earthy", "Earthy T&O", "Steel-slate shadows, ochre/coffee-brown highlights, and muted greens"],
  ["teal_orange_neon", "Neon T&O", "Electric turquoise and glowing tangerine over crisp high-contrast blacks"],
];

const sub = "mb-1 mt-3 text-[10px] uppercase tracking-wider text-neutral-500";
const hint = "mt-1 text-[10px] text-neutral-500";

type Ask = { ask: (prompt: string) => void; aiBusy?: boolean; aiPrompt?: string };

/** the one-click buttons: measured ones run while showing "Working…", ✦ AI ones send a prompt to the AI bar */
function useRunners(p: Ask) {
  const [busy, setBusy] = useState<string | null>(null);
  const run = (key: string, label: ReactNode, f: () => Promise<void>, title: string, className = "", isActive = false) => (
    <Btn
      key={key}
      className={className}
      title={title}
      disabled={busy != null}
      active={busy === key || isActive}
      onClick={async () => {
        setBusy(key);
        try {
          await f();
        } finally {
          setBusy(null);
        }
      }}
    >
      {busy === key ? "Working…" : label}
    </Btn>
  );
  const ai = (label: string, prompt: string, title: string, first?: () => Promise<void>) => {
    const isRunningThis = !!p.aiBusy && p.aiPrompt === prompt;
    return (
      <Btn
        key={label}
        disabled={!!p.aiBusy}
        active={isRunningThis}
        title={`${title} — asks the AI provider; follow it in the AI bar`}
        onClick={async () => (await first?.(), p.ask(prompt))}
      >
        {isRunningThis ? `✦ ${label}…` : `✦ ${label}`}
      </Btn>
    );
  };
  return { run, ai };
}

/** Every automatic adjustment in one place, always for the whole photo. Measured ones run on the photo
 *  and land as one history step each; ✦ AI ones send a prompt to the AI provider through the AI bar.
 *  Auto HDR lives with the rest of HDR in the HDR tab, AI selections in the Masks tab. */
export function AutoTab(
  p: Ask & {
    auto: () => Promise<void>;
    profile: (id: string, label: string, strength?: number) => Promise<void>;
    activeProfile?: string | null;
    profileStrength?: number;
    onStrengthChange?: (strength: number) => void;
    whiteBalance: () => Promise<void>;
    tone: () => Promise<void>;
    levels: () => Promise<void>;
    ca: () => Promise<void>;
    upright: (mode: string) => Promise<void>;
    /** level a tilted horizon (Upright Level) as its own step; ✦ AI fine tune runs it first */
    level: () => Promise<void>;
  },
) {
  const { run, ai } = useRunners(p);
  const [localProfile, setLocalProfile] = useState<string | null>(null);
  const [localStrength, setLocalStrength] = useState<number>(100);

  const activeProfile = p.activeProfile !== undefined ? p.activeProfile : localProfile;
  const strength = p.profileStrength !== undefined ? p.profileStrength : localStrength;

  const onSelect = async (id: string, label: string) => {
    setLocalProfile(id);
    await p.profile(id, label, strength);
  };

  const onStrength = (v: number) => {
    setLocalStrength(v);
    if (p.onStrengthChange) {
      p.onStrengthChange(v);
    } else {
      const profId = activeProfile || "bleach_bypass";
      const profLabel = PROFILES.find(([k]) => k === profId)?.[1] || "Bleach Bypass";
      p.profile(profId, profLabel, v);
    }
  };

  return (
    <>
      <Panel title="Auto fine tune">
        <div className="grid grid-cols-2 gap-1">
          {run("auto", "✦ Auto", p.auto, "White balance and tone measured on the photo")}
          {ai(
            "AI fine tune",
            "Fine tune this photo: natural colour, balanced exposure, no clipped highlights. Keep it realistic. The horizon is levelled separately, so leave crop.angle and transform unchanged.",
            "Level a tilted horizon, then let the AI edit the whole photo",
            p.level,
          )}
        </div>
        <div className={sub}>To a profile</div>
        <div className="grid grid-cols-3 gap-1">
          {PROFILES.map(([id, label, title]) => run(id, label, () => onSelect(id, label), title, "", activeProfile === id))}
        </div>
        <div className="mt-2.5">
          <Slider
            label="Strength"
            value={strength}
            min={0}
            max={200}
            step={1}
            def={100}
            format={(v) => `${v}%`}
            onChange={onStrength}
          />
        </div>
        <p className={hint}>Sets white balance, Basic and presence from the photo, then the profile's look at {strength}% strength. Your other edits stay.</p>
      </Panel>

      <Panel title="One at a time">
        <div className="grid grid-cols-2 gap-1">
          {run("wb", "White balance", p.whiteBalance, "Neutralise a colour cast, judged from the near-grey surfaces")}
          {run("tone", "Tone", p.tone, "Exposure, contrast, highlights, shadows, whites and blacks")}
          {run("levels", "Levels", p.levels, "Each channel's black and white point (Adjust › Levels)")}
          {run("ca", "Chromatic aberration", p.ca, "Line up the colour fringes along edges (Geometry › Lens corrections)")}
        </div>
        <p className={hint}>Auto HDR is in the HDR tab.</p>
      </Panel>

      <Panel title="Auto rotate">
        <div className="grid grid-cols-2 gap-1">
          {run("level", "Straighten", () => p.upright("level"), "Level the horizon from the photo's lines")}
          {run("upright", "Upright", () => p.upright("auto"), "Level and balance the perspective (Geometry › Transform)")}
          {run("vertical", "Verticals", () => p.upright("vertical"), "Level and make vertical lines parallel")}
          {run("full", "Full perspective", () => p.upright("full"), "Level, vertical and horizontal perspective")}
          {ai(
            "Fix orientation",
            "If this photo is sideways or upside down, set image_rotation.angle (clockwise: 90, 180 or 270) so it is upright. If it is already upright, change nothing.",
            "Turn a sideways or upside-down photo upright",
          )}
        </div>
      </Panel>
    </>
  );
}
