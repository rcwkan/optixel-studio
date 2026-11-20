type Rect = { x: number; y: number; w: number; h: number };

/**
 * Corner drag with the aspect ratio locked, all in frame units (0-1). `c0` is the crop when the drag began,
 * `w` and `h` the size the free drag asked for, `k` the crop's w / h (pixel ratio / frame ratio).
 * The corner opposite the handle stays put, the larger of the two moves sets the size, the frame edge caps it.
 */
export function lockCrop(mode: string, c0: Rect, w: number, h: number, k: number): Rect {
  const west = mode.includes("w");
  const north = mode.includes("n");
  const ax = west ? c0.x + c0.w : c0.x;
  const ay = north ? c0.y + c0.h : c0.y;
  const big = Math.max(w, h * k);
  const s = Math.min(1, (west ? ax : 1 - ax) / big, (north ? ay : 1 - ay) / (big / k));
  const nw = big * s;
  const nh = nw / k;
  return { x: Math.max(0, west ? ax - nw : ax), y: Math.max(0, north ? ay - nh : ay), w: nw, h: nh };
}
