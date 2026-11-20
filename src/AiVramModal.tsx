import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { AiVramCheck, api } from "./api";
import { Btn } from "./controls";

const fmtGb = (mb: number | null | undefined) => (mb == null ? "Unknown" : `${(mb / 1024).toFixed(1)} GB`);

/** Pre-start warning dialog when a GPU AI model does not have enough GPU memory with a 20% safety buffer. */
export function AiVramWarningModal({
  check,
  onCancel,
  onContinue,
}: {
  check: AiVramCheck;
  onCancel: () => void;
  onContinue: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 p-4"
      onClick={onCancel}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="ai-vram-warning-title"
        className="w-[460px] max-w-full rounded-lg border border-amber-600/70 bg-neutral-900 p-4 text-[12px] text-neutral-200 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-2.5">
          <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-amber-500/50 bg-amber-500/15 text-xs font-bold text-amber-300">
            !
          </span>
          <div className="min-w-0 flex-1">
            <h2 id="ai-vram-warning-title" className="text-sm font-semibold text-neutral-100">
              Not Enough GPU Memory
            </h2>
            <p className="mt-1 leading-relaxed text-neutral-300">
              {check.message ??
                `${check.model_name} may exceed your available GPU memory (including a 20% buffer) and fall back to slow CPU or shared-memory processing.`}
            </p>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-2 gap-2 rounded border border-neutral-800 bg-neutral-950/70 p-2.5 text-[11px]">
          <div className="min-w-0">
            <div className="text-[9px] uppercase tracking-wider text-neutral-500">Model</div>
            <div className="truncate font-medium text-neutral-200">{check.model_name}</div>
          </div>
          <div className="min-w-0">
            <div className="text-[9px] uppercase tracking-wider text-neutral-500">GPU Device</div>
            <div className="truncate font-medium text-neutral-200">{check.gpu ?? "None (CPU fallback)"}</div>
          </div>
          <div>
            <div className="text-[9px] uppercase tracking-wider text-neutral-500">Available GPU Mem</div>
            <div className="font-mono font-semibold text-amber-300">
              {fmtGb(check.available_vram_mb)}
              {check.total_vram_mb != null && check.available_vram_mb !== check.total_vram_mb
                ? ` / ${fmtGb(check.total_vram_mb)}`
                : ""}
            </div>
          </div>
          <div>
            <div className="text-[9px] uppercase tracking-wider text-neutral-500">Needed (+20% buffer)</div>
            <div className="font-mono font-semibold text-neutral-100">
              {fmtGb(check.required_with_buffer_mb)}
              <span className="ml-1 text-[10px] font-normal text-neutral-500">({fmtGb(check.required_vram_mb)} base)</span>
            </div>
          </div>
        </div>

        <p className="mt-2 text-[10px] text-neutral-400">
          Starting anyway may spill to system RAM or trigger a long CPU pass. You can cancel or proceed below.
        </p>

        <div className="mt-4 flex justify-end gap-2">
          <Btn onClick={onCancel}>Cancel</Btn>
          <Btn active onClick={onContinue}>
            Continue Anyway
          </Btn>
        </div>
      </div>
    </div>,
    document.body,
  );
}

/** Pre-start GPU VRAM check gate with user confirmation modal when free VRAM is below requirement + 20% buffer. */
export function useAiVramGate() {
  const [pending, setPending] = useState<{ check: AiVramCheck; resolve: (ok: boolean) => void } | null>(null);

  const confirmVram = async (modelId: string): Promise<boolean> => {
    try {
      const check = await api.aiCheckVram(modelId);
      if (!check.warning) return true;
      return await new Promise<boolean>((resolve) => {
        setPending({ check, resolve });
      });
    } catch {
      return true;
    }
  };

  const finish = (ok: boolean) => {
    if (!pending) return;
    const { resolve } = pending;
    setPending(null);
    resolve(ok);
  };

  const modal = pending ? (
    <AiVramWarningModal
      check={pending.check}
      onCancel={() => finish(false)}
      onContinue={() => finish(true)}
    />
  ) : null;

  return { confirmVram, modal };
}
