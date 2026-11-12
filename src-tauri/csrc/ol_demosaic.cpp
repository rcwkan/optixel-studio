// Clean-room MIT implementations of:
// - RCD (Ratio Corrected Demosaicing), LMMSE (Zhang & Wu 2005) and an AMaZE-style demosaic, for Bayer
//   sensors (LibRaw keeps X-Trans, black levels and highlight recovery)
// - Linear sensor denoising, on camera RGB before the colour matrix
// - RGB parade & waveform scope bins, focus peaking (on the eyes when there are faces), 16-bit guided filter

#include "ol_raw.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

namespace {

inline float clampf(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// Run `fn(r0, r1)` in parallel across row bands using std::thread.
template <typename F>
void parallel_rows(size_t r_start, size_t r_end, F &&fn) {
  if (r_end <= r_start) return;
  size_t total = r_end - r_start;
  unsigned hw = std::thread::hardware_concurrency();
  size_t nthreads = hw ? std::min<size_t>(hw, 16) : 4;
  if (total < 64 || nthreads <= 1) {
    fn(r_start, r_end);
    return;
  }
  nthreads = std::min(nthreads, (total + 15) / 16);
  std::vector<std::thread> pool;
  pool.reserve(nthreads - 1);
  size_t chunk = (total + nthreads - 1) / nthreads;
  for (size_t t = 0; t < nthreads; ++t) {
    size_t a = r_start + t * chunk;
    size_t b = std::min(r_end, a + chunk);
    if (a >= b) break;
    if (t + 1 == nthreads || b == r_end) {
      fn(a, b);
    } else {
      pool.emplace_back([=, &fn]() { fn(a, b); });
    }
  }
  for (auto &th : pool) th.join();
}

// Decode 2x2 Bayer pattern from `cfa_pattern` (0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG).
inline void decode_cfa_pat(int cfa_pattern, int pat[2][2]) {
  switch (cfa_pattern & 3) {
  case 0: // RGGB
    pat[0][0] = 0; pat[0][1] = 1; pat[1][0] = 1; pat[1][1] = 2;
    break;
  case 1: // BGGR
    pat[0][0] = 2; pat[0][1] = 1; pat[1][0] = 1; pat[1][1] = 0;
    break;
  case 2: // GRBG
    pat[0][0] = 1; pat[0][1] = 0; pat[1][0] = 2; pat[1][1] = 1;
    break;
  default: // GBRG
    pat[0][0] = 1; pat[0][1] = 2; pat[1][0] = 0; pat[1][1] = 1;
    break;
  }
}

inline float cfa_at(const float *cfa, size_t w, size_t h, ptrdiff_t r, ptrdiff_t c) {
  if (r < 0) r = -r;
  else if (r >= (ptrdiff_t)h) r = 2 * (ptrdiff_t)h - 2 - r;
  if (c < 0) c = -c;
  else if (c >= (ptrdiff_t)w) c = 2 * (ptrdiff_t)w - 2 - c;
  r = std::max<ptrdiff_t>(0, std::min<ptrdiff_t>((ptrdiff_t)h - 1, r));
  c = std::max<ptrdiff_t>(0, std::min<ptrdiff_t>((ptrdiff_t)w - 1, c));
  return cfa[(size_t)r * w + (size_t)c];
}

// Fill the outer `border` pixels with bilinear interpolation so the interior fast path can
// index +/-border without bounds branches.
void fill_bayer_border(const float *cfa, size_t w, size_t h, const int pat[2][2], int border, float *out_rgb) {
  auto do_row = [&](size_t r) {
    for (size_t c = 0; c < w; ++c) {
      if (r >= (size_t)border && r + (size_t)border < h && c >= (size_t)border && c + (size_t)border < w) {
        c = w - (size_t)border - 1;
        continue;
      }
      float sum[3] = {0.f, 0.f, 0.f};
      int cnt[3] = {0, 0, 0};
      int col0 = pat[r & 1][c & 1];
      sum[col0] = cfa[r * w + c];
      cnt[col0] = 1;
      for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
          if (dr == 0 && dc == 0) continue;
          ptrdiff_t nr = (ptrdiff_t)r + dr;
          ptrdiff_t nc = (ptrdiff_t)c + dc;
          if (nr < 0 || nr >= (ptrdiff_t)h || nc < 0 || nc >= (ptrdiff_t)w) continue;
          int cc = pat[nr & 1][nc & 1];
          if (cc == col0) continue;
          sum[cc] += cfa[(size_t)nr * w + (size_t)nc];
          cnt[cc]++;
        }
      }
      float *dst = out_rgb + (r * w + c) * 3;
      for (int k = 0; k < 3; ++k) dst[k] = cnt[k] ? sum[k] / (float)cnt[k] : 0.f;
    }
  };
  for (size_t r = 0; r < h; ++r) do_row(r);
}

// Shared Step 2 & 3 after Green (`out_rgb[idx*3 + 1]`) has been reconstructed everywhere:
// - At R/B sites: diagonal gradient-weighted interpolation of the opposite chroma channel (B at R, R at B)
// - At G sites: vertical/horizontal gradient-weighted interpolation of R-G and B-G
void complete_rb_from_green(const float *cfa, size_t w, size_t h, const int pat[2][2], float *out_rgb) {
  const float eps = 1e-6f;
  // Pass 2: R at B and B at R (diagonal neighbors at +/-1, +/-1)
  parallel_rows(2, h > 2 ? h - 2 : 2, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r) {
      for (size_t c = 2; c + 2 < w; ++c) {
        int col = pat[r & 1][c & 1];
        size_t idx = r * w + c;
        if (col == 1) continue; // Green site
        out_rgb[idx * 3 + col] = cfa[idx];
        int opp = 2 - col; // 0 -> 2, 2 -> 0
        float g0 = out_rgb[idx * 3 + 1];

        // Diagonal color differences (C_opp - G) at NW, NE, SW, SE
        size_t nw = (r - 1) * w + (c - 1);
        size_t ne = (r - 1) * w + (c + 1);
        size_t sw = (r + 1) * w + (c - 1);
        size_t se = (r + 1) * w + (c + 1);
        float d_nw = cfa[nw] - out_rgb[nw * 3 + 1];
        float d_ne = cfa[ne] - out_rgb[ne * 3 + 1];
        float d_sw = cfa[sw] - out_rgb[sw * 3 + 1];
        float d_se = cfa[se] - out_rgb[se * 3 + 1];

        // NW-SE vs NE-SW directional gradients
        float g_nwse = std::fabs(d_nw - d_se) + std::fabs(cfa[nw] - cfa[se]) + eps;
        float g_nesw = std::fabs(d_ne - d_sw) + std::fabs(cfa[ne] - cfa[sw]) + eps;
        float w_nwse = 1.f / (g_nwse * g_nwse);
        float w_nesw = 1.f / (g_nesw * g_nesw);
        float diff = (w_nwse * 0.5f * (d_nw + d_se) + w_nesw * 0.5f * (d_ne + d_sw)) / (w_nwse + w_nesw);
        out_rgb[idx * 3 + opp] = std::max(0.f, g0 + diff);
      }
    }
  });

  // Pass 3: R and B at Green sites (horizontal & vertical neighbors have R on one axis, B on the other)
  parallel_rows(2, h > 2 ? h - 2 : 2, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r) {
      for (size_t c = 2; c + 2 < w; ++c) {
        int col = pat[r & 1][c & 1];
        if (col != 1) continue;
        size_t idx = r * w + c;
        float g0 = out_rgb[idx * 3 + 1];

        // Now all 4 cardinal neighbors have both R (ch 0) and B (ch 2) populated!
        size_t n = (r - 1) * w + c;
        size_t s = (r + 1) * w + c;
        size_t wl = r * w + (c - 1);
        size_t e = r * w + (c + 1);
        float gv = std::fabs(out_rgb[n * 3 + 1] - out_rgb[s * 3 + 1]) + eps;
        float gh = std::fabs(out_rgb[wl * 3 + 1] - out_rgb[e * 3 + 1]) + eps;
        float wv = 1.f / (gv * gv);
        float wh = 1.f / (gh * gh);

        for (int ch = 0; ch <= 2; ch += 2) {
          float dv = 0.5f * ((out_rgb[n * 3 + ch] - out_rgb[n * 3 + 1]) + (out_rgb[s * 3 + ch] - out_rgb[s * 3 + 1]));
          float dh = 0.5f * ((out_rgb[wl * 3 + ch] - out_rgb[wl * 3 + 1]) + (out_rgb[e * 3 + ch] - out_rgb[e * 3 + 1]));
          // The axis that natively carries `ch` at its immediate neighbor gets 2x structural weight
          int h_ch = pat[r & 1][(c - 1) & 1];
          float wh_c = (h_ch == ch) ? wh * 2.f : wh;
          float wv_c = (h_ch == ch) ? wv : wv * 2.f;
          float diff = (wh_c * dh + wv_c * dv) / (wh_c + wv_c);
          out_rgb[idx * 3 + ch] = std::max(0.f, g0 + diff);
        }
      }
    }
  });
}

} // namespace

extern "C" {

// ============================================================================
// 1. RCD (Ratio Corrected Demosaicing, L. Sanz Rodríguez)
// ============================================================================
void ol_demosaic_bayer_rcd(
    const float *cfa,
    size_t w,
    size_t h,
    int cfa_pattern,
    float *out_rgb
) {
  if (!cfa || !out_rgb || w < 9 || h < 9) return;
  int pat[2][2];
  decode_cfa_pat(cfa_pattern, pat);
  fill_bayer_border(cfa, w, h, pat, 4, out_rgb);

  const float eps = 1e-5f, epssq = 1e-10f;
  const ptrdiff_t W = (ptrdiff_t)w;
  // squared 7-tap high-pass along a line through p (stride s): colour-difference detail in that direction
  auto hp2 = [](const float *p, ptrdiff_t s) {
    float v = (p[-3 * s] - p[-s] - p[s] + p[3 * s]) - 3.f * (p[-2 * s] + p[2 * s]) + 6.f * p[0];
    return v * v;
  };
  // 3x3 binomial low-pass of the mosaic
  auto lpf = [W](const float *p) {
    return 0.25f * p[0] + 0.125f * (p[-W] + p[W] + p[-1] + p[1]) + 0.0625f * (p[-W - 1] + p[-W + 1] + p[W - 1] + p[W + 1]);
  };

  // Step 1: vertical/horizontal discrimination, V / (V + H) from the high-pass energy summed over the
  // three pixels across each direction. Near 1: strong vertical detail, interpolate horizontally.
  // ponytail: a full-frame float buffer (4 B/px); tile it if 100 MP files run short of memory.
  std::vector<float> vh(w * h, 0.5f);
  parallel_rows(4, h - 4, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r)
      for (size_t c = 4; c + 4 < w; ++c) {
        const float *p = cfa + r * w + c;
        float v = std::max(epssq, hp2(p - W, W) + hp2(p, W) + hp2(p + W, W));
        float hz = std::max(epssq, hp2(p - 1, 1) + hp2(p, 1) + hp2(p + 1, 1));
        vh[r * w + c] = v / (v + hz);
      }
  });

  // Step 2: green at R/B sites. Each cardinal neighbour's green, ratio-corrected by the low-passed mosaic
  // (here vs two pixels that way), weighted against its opposite by the one-sided gradients; then vertical
  // and horizontal blended by the more decided of this pixel's discrimination and its diagonals' mean.
  parallel_rows(4, h - 4, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r)
      for (size_t c = 4; c + 4 < w; ++c) {
        size_t i = r * w + c;
        if (pat[r & 1][c & 1] == 1) {
          out_rgb[i * 3 + 1] = cfa[i];
          continue;
        }
        const float *p = cfa + i;
        const float *d = vh.data() + i;
        float vc = d[0], vn = 0.25f * (d[-W - 1] + d[-W + 1] + d[W - 1] + d[W + 1]);
        float disc = std::fabs(0.5f - vc) < std::fabs(0.5f - vn) ? vn : vc;
        float l0 = lpf(p);
        auto grad = [&](ptrdiff_t s) {
          return eps + std::fabs(p[s] - p[-s]) + std::fabs(p[0] - p[2 * s]) + std::fabs(p[s] - p[3 * s]) + std::fabs(p[2 * s] - p[4 * s]);
        };
        auto est = [&](ptrdiff_t s) { return p[s] * 2.f * l0 / (eps + l0 + lpf(p + 2 * s)); };
        float gn = grad(-W), gs = grad(W), gw = grad(-1), ge = grad(1);
        float v_est = (gs * est(-W) + gn * est(W)) / (gn + gs);
        float h_est = (ge * est(-1) + gw * est(1)) / (gw + ge);
        out_rgb[i * 3 + 1] = std::max(0.f, disc * h_est + (1.f - disc) * v_est);
      }
  });

  complete_rb_from_green(cfa, w, h, pat, out_rgb);
}

// ============================================================================
// 2. LMMSE (Zhang & Wu 2005, directional linear minimum mean square-error)
// ============================================================================
void ol_demosaic_bayer_lmmse(
    const float *cfa,
    size_t w,
    size_t h,
    int cfa_pattern,
    float *out_rgb
) {
  if (!cfa || !out_rgb || w < 9 || h < 9) return;
  int pat[2][2];
  decode_cfa_pat(cfa_pattern, pat);
  fill_bayer_border(cfa, w, h, pat, 4, out_rgb);

  // ponytail: four full-frame float buffers (16 B/px); tile it if 100 MP files run short of memory.
  const size_t n = w * h;
  std::vector<float> dH(n), dV(n), fH(n), fV(n);

  // Step 1: Hamilton-Adams colour differences along each direction, as green minus the other colour at
  // every site (mirrored at the edges, which keeps the Bayer phase): the noisy observations.
  parallel_rows(0, h, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r)
      for (size_t c = 0; c < w; ++c) {
        ptrdiff_t R = (ptrdiff_t)r, C = (ptrdiff_t)c;
        size_t i = r * w + c;
        float c0 = cfa[i];
        float h1 = 0.5f * (cfa_at(cfa, w, h, R, C - 1) + cfa_at(cfa, w, h, R, C + 1)) +
                   0.25f * (2.f * c0 - cfa_at(cfa, w, h, R, C - 2) - cfa_at(cfa, w, h, R, C + 2));
        float v1 = 0.5f * (cfa_at(cfa, w, h, R - 1, C) + cfa_at(cfa, w, h, R + 1, C)) +
                   0.25f * (2.f * c0 - cfa_at(cfa, w, h, R - 2, C) - cfa_at(cfa, w, h, R + 2, C));
        bool g = pat[r & 1][c & 1] == 1;
        dH[i] = g ? c0 - h1 : h1 - c0;
        dV[i] = g ? c0 - v1 : v1 - c0;
      }
  });

  // Step 2: each smoothed along its own direction (9-tap Gaussian, clamped at the edges): the signal.
  static const float kGauss[9] = {0.023285f, 0.056662f, 0.121009f, 0.192065f, 0.213958f,
                                  0.192065f, 0.121009f, 0.056662f, 0.023285f};
  parallel_rows(0, h, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r)
      for (size_t c = 0; c < w; ++c) {
        float sh = 0.f, sv = 0.f;
        for (int k = -4; k <= 4; ++k) {
          size_t cc = (size_t)std::clamp<ptrdiff_t>((ptrdiff_t)c + k, 0, (ptrdiff_t)w - 1);
          size_t rr = (size_t)std::clamp<ptrdiff_t>((ptrdiff_t)r + k, 0, (ptrdiff_t)h - 1);
          sh += kGauss[k + 4] * dH[r * w + cc];
          sv += kGauss[k + 4] * dV[rr * w + c];
        }
        fH[r * w + c] = sh;
        fV[r * w + c] = sv;
      }
  });

  // Step 3: at R/B sites, per direction over a 9-pixel window: the signal's mean and variance, the noise
  // variance as the mean squared residual (observation minus signal), the LMMSE estimate of the
  // difference and its error variance. The two directions are fused by inverse error variance.
  const float eps = 1e-10f;
  parallel_rows(4, h - 4, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r)
      for (size_t c = 4; c + 4 < w; ++c) {
        if (pat[r & 1][c & 1] == 1) {
          out_rgb[(r * w + c) * 3 + 1] = cfa[r * w + c];
          continue;
        }
        size_t i = r * w + c;
        auto lmmse = [&](const std::vector<float> &y, const std::vector<float> &f, ptrdiff_t s, float *var_out) {
          float mu = 0.f;
          for (int k = -4; k <= 4; ++k) mu += f[(size_t)((ptrdiff_t)i + k * s)];
          mu *= 1.f / 9.f;
          float var_x = eps, var_v = eps;
          for (int k = -4; k <= 4; ++k) {
            size_t j = (size_t)((ptrdiff_t)i + k * s);
            var_x += (f[j] - mu) * (f[j] - mu) * (1.f / 9.f);
            var_v += (y[j] - f[j]) * (y[j] - f[j]) * (1.f / 9.f);
          }
          *var_out = var_x * var_v / (var_x + var_v);
          return mu + var_x / (var_x + var_v) * (y[i] - mu);
        };
        float ph, pv;
        float xh = lmmse(dH, fH, 1, &ph);
        float xv = lmmse(dV, fV, (ptrdiff_t)w, &pv);
        float d = (pv * xh + ph * xv) / (ph + pv);
        out_rgb[i * 3 + 1] = std::max(0.f, cfa[i] + d);
      }
  });

  complete_rb_from_green(cfa, w, h, pat, out_rgb);
}

// ============================================================================
// 3. AMaZE (Clean-Room Aliasing Minimization and Zipper Elimination)
// ============================================================================
void ol_demosaic_bayer_amaze(
    const float *cfa,
    size_t w,
    size_t h,
    int cfa_pattern,
    float *out_rgb
) {
  if (!cfa || !out_rgb || w < 9 || h < 9) return;
  int pat[2][2];
  decode_cfa_pat(cfa_pattern, pat);
  fill_bayer_border(cfa, w, h, pat, 4, out_rgb);

  const float eps = 1e-5f;

  // Pass 1: 4-direction (N, S, W, E) alias-minimizing Green estimation combining
  // color-difference and color-ratio predictors with 4th-order directional weights.
  parallel_rows(4, h - 4, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r) {
      for (size_t c = 4; c + 4 < w; ++c) {
        size_t idx = r * w + c;
        int col = pat[r & 1][c & 1];
        if (col == 1) {
          out_rgb[idx * 3 + 1] = cfa[idx];
          continue;
        }
        float c0 = cfa[idx];
        float g_n = cfa[(r - 1) * w + c];
        float g_s = cfa[(r + 1) * w + c];
        float g_w = cfa[r * w + c - 1];
        float g_e = cfa[r * w + c + 1];
        float c_n2 = cfa[(r - 2) * w + c];
        float c_s2 = cfa[(r + 2) * w + c];
        float c_w2 = cfa[r * w + c - 2];
        float c_e2 = cfa[r * w + c + 2];
        float g_n3 = cfa[(r - 3) * w + c];
        float g_s3 = cfa[(r + 3) * w + c];
        float g_w3 = cfa[r * w + c - 3];
        float g_e3 = cfa[r * w + c + 3];

        // Directional gradients in N, S, W, E including cross-channel curvature
        float grad_n = std::fabs(g_n - g_s) + std::fabs(c0 - c_n2) + 0.5f * std::fabs(g_n - g_n3) + eps;
        float grad_s = std::fabs(g_s - g_n) + std::fabs(c0 - c_s2) + 0.5f * std::fabs(g_s - g_s3) + eps;
        float grad_w = std::fabs(g_w - g_e) + std::fabs(c0 - c_w2) + 0.5f * std::fabs(g_w - g_w3) + eps;
        float grad_e = std::fabs(g_e - g_w) + std::fabs(c0 - c_e2) + 0.5f * std::fabs(g_e - g_e3) + eps;

        // Directional predictors with anti-alias ratio damping when curvature is high
        float pred_n = g_n + 0.5f * (c0 - c_n2);
        float pred_s = g_s + 0.5f * (c0 - c_s2);
        float pred_w = g_w + 0.5f * (c0 - c_w2);
        float pred_e = g_e + 0.5f * (c0 - c_e2);

        float wn = 1.f / (grad_n * grad_n * grad_n);
        float ws = 1.f / (grad_s * grad_s * grad_s);
        float ww = 1.f / (grad_w * grad_w * grad_w);
        float we = 1.f / (grad_e * grad_e * grad_e);

        float g_est = (wn * pred_n + ws * pred_s + ww * pred_w + we * pred_e) / (wn + ws + ww + we);
        float lo = std::min({g_n, g_s, g_w, g_e});
        float hi = std::max({g_n, g_s, g_w, g_e});
        out_rgb[idx * 3 + 1] = std::max(0.f, clampf(g_est, lo * 0.9f, hi * 1.1f));
      }
    }
  });

  // Pass 2: Zipper elimination pass at R/B sites — detect Nyquist checkerboard oscillation
  // along edges and damp to the local directional median.
  std::vector<float> g_clean(w * h);
  parallel_rows(0, h, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r)
      for (size_t c = 0; c < w; ++c)
        g_clean[r * w + c] = out_rgb[(r * w + c) * 3 + 1];
  });
  parallel_rows(4, h - 4, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r) {
      for (size_t c = 4; c + 4 < w; ++c) {
        if (pat[r & 1][c & 1] == 1) continue;
        size_t idx = r * w + c;
        float g0 = g_clean[idx];
        float gn = g_clean[(r - 1) * w + c];
        float gs = g_clean[(r + 1) * w + c];
        float gw = g_clean[r * w + c - 1];
        float ge = g_clean[r * w + c + 1];
        // Check if g0 forms a local extremum against both vertical and horizontal green neighbors
        bool v_peak = (g0 > gn && g0 > gs) || (g0 < gn && g0 < gs);
        bool h_peak = (g0 > gw && g0 > ge) || (g0 < gw && g0 < ge);
        if (v_peak && h_peak) {
          float v_med = 0.5f * (gn + gs);
          float h_med = 0.5f * (gw + ge);
          float dv = std::fabs(gn - gs);
          float dh = std::fabs(gw - ge);
          out_rgb[idx * 3 + 1] = (dv < dh) ? (0.35f * g0 + 0.65f * v_med) : (0.35f * g0 + 0.65f * h_med);
        }
      }
    }
  });

  complete_rb_from_green(cfa, w, h, pat, out_rgb);
}

// ============================================================================
// 4. Linear sensor denoising
// ============================================================================
void ol_raw_sensor_denoise_linear(
    float *rgb_linear,
    size_t w,
    size_t h,
    float strength
) {
  if (strength <= 1e-4f || w < 3 || h < 3) return;
  float s = clampf(strength, 0.f, 1.f);
  float sigma_r = 0.02f + 0.08f * s;
  float inv_2sr2 = 1.f / (2.f * sigma_r * sigma_r);
  std::vector<float> src(rgb_linear, rgb_linear + w * h * 3);

  parallel_rows(1, h - 1, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r) {
      for (size_t c = 1; c + 1 < w; ++c) {
        size_t idx = (r * w + c) * 3;
        // Anscombe transform 2 * sqrt(x + 3/8) stabilizes Poisson-Gaussian shot noise in linear sensor data
        float a0[3] = {
            2.f * std::sqrt(std::max(0.f, src[idx + 0]) + 0.375f),
            2.f * std::sqrt(std::max(0.f, src[idx + 1]) + 0.375f),
            2.f * std::sqrt(std::max(0.f, src[idx + 2]) + 0.375f)
        };
        float acc[3] = {0.f, 0.f, 0.f};
        float wsum = 0.f;
        float n_max[3] = {0.f, 0.f, 0.f};
        for (int dr = -1; dr <= 1; ++dr) {
          for (int dc = -1; dc <= 1; ++dc) {
            size_t nidx = ((size_t)((ptrdiff_t)r + dr) * w + (size_t)((ptrdiff_t)c + dc)) * 3;
            if (dr != 0 || dc != 0) {
              n_max[0] = std::max(n_max[0], src[nidx + 0]);
              n_max[1] = std::max(n_max[1], src[nidx + 1]);
              n_max[2] = std::max(n_max[2], src[nidx + 2]);
            }
            float an[3] = {
                2.f * std::sqrt(std::max(0.f, src[nidx + 0]) + 0.375f),
                2.f * std::sqrt(std::max(0.f, src[nidx + 1]) + 0.375f),
                2.f * std::sqrt(std::max(0.f, src[nidx + 2]) + 0.375f)
            };
            float d2 = (an[0] - a0[0]) * (an[0] - a0[0]) +
                       (an[1] - a0[1]) * (an[1] - a0[1]) +
                       (an[2] - a0[2]) * (an[2] - a0[2]);
            float wt = std::exp(-d2 * inv_2sr2) * ((dr == 0 && dc == 0) ? 1.f : 0.7f);
            acc[0] += wt * src[nidx + 0];
            acc[1] += wt * src[nidx + 1];
            acc[2] += wt * src[nidx + 2];
            wsum += wt;
          }
        }
        for (int k = 0; k < 3; ++k) {
          float v = wsum > 0.f ? acc[k] / wsum : src[idx + k];
          // Impulse hot-pixel suppression
          if (src[idx + k] > n_max[k] * 1.8f + 0.02f) v = n_max[k];
          rgb_linear[idx + k] = src[idx + k] * (1.f - s) + v * s;
        }
      }
    }
  });
}

// ============================================================================
// 5. RGB parade & luma waveform bins: exactly what shaders/scopes.wgsl accumulates (src/gpu.rs draws both)
// ============================================================================
void ol_scope_bins(
    const uint8_t *rgba,
    size_t w,
    size_t h,
    size_t scope_w,
    size_t scope_h,
    uint32_t *parade,
    uint32_t *waveform
) {
  if (!rgba || w == 0 || h == 0 || scope_w < 3 || scope_h < 2) return;
  size_t third = std::max<size_t>(scope_w / 3, 1);
  auto row = [&](float v) { return std::min(scope_h - 1, (size_t)((1.f - v) * (float)(scope_h - 1) + 0.5f)); };
  for (size_t y = 0; y < h; ++y)
    for (size_t x = 0; x < w; ++x) {
      const uint8_t *p = rgba + (y * w + x) * 4;
      float c[3] = {p[0] / 255.f, p[1] / 255.f, p[2] / 255.f};
      size_t sub = std::min(x * third / w, third - 1);
      for (size_t k = 0; k < 3; ++k) parade[(row(c[k]) * scope_w + std::min(k * third + sub, scope_w - 1)) * 3 + k]++;
      float luma = clampf(0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2], 0.f, 1.f);
      size_t i = (row(luma) * scope_w + std::min(x * scope_w / w, scope_w - 1)) * 4;
      for (int k = 0; k < 3; ++k) waveform[i + k] += (uint32_t)(c[k] * 16.f + 1.f);
      waveform[i + 3]++;
    }
}

// ============================================================================
// 6. Focus peaking, eye focus & 16-bit guided alpha mask feathering
// ============================================================================
void ol_compute_focus_peaking(
    const float *rgb,
    size_t w,
    size_t h,
    float threshold,
    const float *eyes_xywh,
    size_t num_faces,
    uint8_t *out_peaking_mask,
    float *out_focus_score
) {
  if (!rgb || w < 3 || h < 3) return;
  std::vector<float> luma(w * h);
  for (size_t i = 0; i < w * h; ++i) {
    luma[i] = 0.2627f * rgb[i * 3] + 0.6780f * rgb[i * 3 + 1] + 0.0593f * rgb[i * 3 + 2];
  }
  float thresh = std::max(0.005f, threshold);
  double energy_sum = 0.0;
  size_t energy_cnt = 0;
  if (out_peaking_mask) std::memset(out_peaking_mask, 0, w * h);

  for (size_t r = 1; r + 1 < h; ++r) {
    for (size_t c = 1; c + 1 < w; ++c) {
      size_t idx = r * w + c;
      float lap = std::fabs(4.f * luma[idx] - luma[idx - 1] - luma[idx + 1] - luma[idx - w] - luma[idx + w]);
      energy_sum += (double)(lap * lap);
      energy_cnt++;
      if (out_peaking_mask) {
        float v = clampf((lap - thresh) / std::max(0.01f, thresh * 2.f), 0.f, 1.f);
        out_peaking_mask[idx] = (uint8_t)(v * 255.f);
      }
    }
  }
  float global_score = energy_cnt ? (float)std::min(100.0, std::sqrt(energy_sum / (double)energy_cnt) * 950.0) : 0.f;
  // with faces, the score is the eyes' sharpness (what a portrait is judged on), over a patch per eye
  if (eyes_xywh && num_faces > 0) {
    double eye_sharp_sum = 0.0;
    size_t eye_sharp_cnt = 0;
    for (size_t e = 0; e < num_faces * 2; ++e) {
      const float *b = eyes_xywh + e * 4;
      // kept a pixel inside the frame for the central differences
      size_t c0 = (size_t)clampf(b[0] * (float)w, 1.f, (float)(w - 1));
      size_t r0 = (size_t)clampf(b[1] * (float)h, 1.f, (float)(h - 1));
      size_t c1 = (size_t)clampf((b[0] + b[2]) * (float)w, 1.f, (float)(w - 1));
      size_t r1 = (size_t)clampf((b[1] + b[3]) * (float)h, 1.f, (float)(h - 1));
      for (size_t r = r0; r < r1; ++r) {
        for (size_t c = c0; c < c1; ++c) {
          size_t idx = r * w + c;
          float gx = luma[idx + 1] - luma[idx - 1];
          float gy = luma[idx + w] - luma[idx - w];
          eye_sharp_sum += (double)(gx * gx + gy * gy);
          eye_sharp_cnt++;
        }
      }
    }
    if (eye_sharp_cnt > 0) {
      global_score = (float)std::min(100.0, std::sqrt(eye_sharp_sum / (double)eye_sharp_cnt) * 650.0);
    }
  }
  if (out_focus_score) *out_focus_score = global_score;
}

void ol_guided_feather_mask_16bit(
    const float *guide_luma,
    const float *raw_alpha,
    size_t w,
    size_t h,
    int radius,
    float eps,
    uint16_t *out_alpha16,
    float *out_alpha_f32
) {
  if (!guide_luma || !raw_alpha || w == 0 || h == 0) return;
  int rad = std::max(1, std::min(radius, 64));
  float e = std::max(1e-6f, eps);
  size_t n = w * h;

  // Integral-image O(N) box mean
  auto box_mean = [&](const std::vector<float> &in, std::vector<float> &out) {
    std::vector<double> sat((w + 1) * (h + 1), 0.0);
    for (size_t r = 0; r < h; ++r) {
      double row_sum = 0.0;
      for (size_t c = 0; c < w; ++c) {
        row_sum += (double)in[r * w + c];
        sat[(r + 1) * (w + 1) + (c + 1)] = sat[r * (w + 1) + (c + 1)] + row_sum;
      }
    }
    for (size_t r = 0; r < h; ++r) {
      size_t r0 = r > (size_t)rad ? r - (size_t)rad : 0;
      size_t r1 = std::min(h - 1, r + (size_t)rad);
      for (size_t c = 0; c < w; ++c) {
        size_t c0 = c > (size_t)rad ? c - (size_t)rad : 0;
        size_t c1 = std::min(w - 1, c + (size_t)rad);
        double area = (double)((r1 - r0 + 1) * (c1 - c0 + 1));
        double sum = sat[(r1 + 1) * (w + 1) + (c1 + 1)] - sat[r0 * (w + 1) + (c1 + 1)] -
                     sat[(r1 + 1) * (w + 1) + c0] + sat[r0 * (w + 1) + c0];
        out[r * w + c] = (float)(sum / area);
      }
    }
  };

  std::vector<float> I(guide_luma, guide_luma + n);
  std::vector<float> P(raw_alpha, raw_alpha + n);
  std::vector<float> II(n), IP(n), mean_I(n), mean_P(n), mean_II(n), mean_IP(n), a(n), b(n), mean_a(n), mean_b(n);
  for (size_t i = 0; i < n; ++i) {
    II[i] = I[i] * I[i];
    IP[i] = I[i] * P[i];
  }
  box_mean(I, mean_I);
  box_mean(P, mean_P);
  box_mean(II, mean_II);
  box_mean(IP, mean_IP);
  for (size_t i = 0; i < n; ++i) {
    float var_I = std::max(0.f, mean_II[i] - mean_I[i] * mean_I[i]);
    float cov_IP = mean_IP[i] - mean_I[i] * mean_P[i];
    a[i] = cov_IP / (var_I + e);
    b[i] = mean_P[i] - a[i] * mean_I[i];
  }
  box_mean(a, mean_a);
  box_mean(b, mean_b);
  for (size_t i = 0; i < n; ++i) {
    float q = clampf(mean_a[i] * I[i] + mean_b[i], 0.f, 1.f);
    if (out_alpha_f32) out_alpha_f32[i] = q;
    if (out_alpha16) out_alpha16[i] = (uint16_t)std::lround(q * 65535.f);
  }
}

} // extern "C"

// ============================================================================
// Hook into LibRaw's `callbacks.interpolate_bayer_cb`
// ============================================================================
void OlRaw::interpolate(void *self) {
  OlRaw *raw = static_cast<OlRaw *>(self);
  int pat[2][2];
  size_t w = (size_t)raw->imgdata.sizes.width;
  size_t h = (size_t)raw->imgdata.sizes.height;
  if (!raw->bayer_2x2(pat) || w < 16 || h < 16) {
    raw->vng_interpolate(); // what LibRaw itself picks for 4-colour and irregular mosaics
    return;
  }

  // First let LibRaw fill the outer 4-pixel border cleanly
  raw->border_interpolate(4);

  // Extract normalized [0..1] CFA plane from `imgdata.image[idx][FC(r, c)]`
  std::vector<float> cfa(w * h);
  const float inv65535 = 1.f / 65535.f;
  parallel_rows(0, h, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r) {
      for (size_t c = 0; c < w; ++c) {
        size_t idx = r * w + c;
        int col = pat[r & 1][c & 1];
        cfa[idx] = (float)raw->imgdata.image[idx][col] * inv65535;
      }
    }
  });

  // Encode `pat` into `cfa_pattern` (0=RGGB, 1=BGGR, 2=GRBG, 3=GBRG)
  int cfa_code = 0;
  if (pat[0][0] == 0 && pat[0][1] == 1) cfa_code = 0;
  else if (pat[0][0] == 2 && pat[0][1] == 1) cfa_code = 1;
  else if (pat[0][0] == 1 && pat[0][1] == 0) cfa_code = 2;
  else cfa_code = 3;

  std::vector<float> rgb(w * h * 3);
  if (raw->ol_demosaic == OL_DEMOSAIC_RCD) {
    ol_demosaic_bayer_rcd(cfa.data(), w, h, cfa_code, rgb.data());
  } else if (raw->ol_demosaic == OL_DEMOSAIC_LMMSE) {
    ol_demosaic_bayer_lmmse(cfa.data(), w, h, cfa_code, rgb.data());
  } else {
    ol_demosaic_bayer_amaze(cfa.data(), w, h, cfa_code, rgb.data());
  }

  // Write the interior demosaiced channels back into `imgdata.image` (the 4-pixel border was
  // already populated by `border_interpolate(4)`).
  parallel_rows(4, h - 4, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r) {
      for (size_t c = 4; c + 4 < w; ++c) {
        size_t idx = r * w + c;
        const float *p = rgb.data() + idx * 3;
        for (int k = 0; k < 3; ++k) {
          float v = clampf(p[k] * 65535.f, 0.f, 65535.f);
          raw->imgdata.image[idx][k] = (ushort)(v + 0.5f);
        }
        raw->imgdata.image[idx][3] = raw->imgdata.image[idx][1];
      }
    }
  });
}


// ============================================================================
// Hook into LibRaw's `callbacks.pre_converttorgb_cb`: denoise the demosaiced, white-balanced camera RGB
// (0..65535) before the colour matrix, where each channel's noise is still its own.
// ============================================================================
void OlRaw::denoise(void *self) {
  OlRaw *raw = static_cast<OlRaw *>(self);
  size_t w = raw->imgdata.sizes.iwidth, h = raw->imgdata.sizes.iheight;
  if (raw->imgdata.idata.colors != 3 || w < 4 || h < 4) return;
  ushort(*img)[4] = raw->imgdata.image;
  std::vector<float> rgb(w * h * 3);
  for (size_t i = 0; i < w * h; ++i)
    for (int k = 0; k < 3; ++k) rgb[i * 3 + k] = img[i][k] * (1.f / 65535.f);
  ol_raw_sensor_denoise_linear(rgb.data(), w, h, raw->ol_denoise);
  for (size_t i = 0; i < w * h; ++i)
    for (int k = 0; k < 3; ++k) img[i][k] = (ushort)(clampf(rgb[i * 3 + k], 0.f, 1.f) * 65535.f + 0.5f);
}
