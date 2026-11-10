// Clean-room MIT implementations of:
// - RCD (Ratio Corrected Demosaicing)
// - LMMSE (Zhang & Wu 2005 Directional Linear Minimum Mean Square Error Demosaicing)
// - AMaZE (Clean-room Aliasing Minimization & Zipper Elimination Demosaicing)
// - X-Trans 6x6 directional demosaicing
// - Optical black-point calibration, highlight reconstruction & linear sensor denoising
// - Real-time RGB Parade & Waveform scopes, Focus-peaking & Blink EAR, 16-bit Guided Filter

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
// 1. RCD (Ratio Corrected Demosaicing)
// ============================================================================
void ol_demosaic_bayer_rcd(
    const float *cfa,
    size_t w,
    size_t h,
    int cfa_pattern,
    float *out_rgb
) {
  int pat[2][2];
  decode_cfa_pat(cfa_pattern, pat);
  fill_bayer_border(cfa, w, h, pat, 4, out_rgb);
  if (w < 9 || h < 9) return;

  const float eps = 1e-5f;
  const float epssq = 1e-10f;

  // Low-pass ratio estimate P_V and P_H on a per-row basis + directional V_stat / H_stat
  parallel_rows(4, h - 4, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r) {
      for (size_t c = 4; c + 4 < w; ++c) {
        size_t idx = r * w + c;
        int col = pat[r & 1][c & 1];
        if (col == 1) {
          out_rgb[idx * 3 + 1] = cfa[idx];
          continue;
        }
        float c0 = std::max(eps, cfa[idx]);
        float c_n1 = std::max(eps, cfa[(r - 1) * w + c]);
        float c_s1 = std::max(eps, cfa[(r + 1) * w + c]);
        float c_w1 = std::max(eps, cfa[r * w + (c - 1)]);
        float c_e1 = std::max(eps, cfa[r * w + (c + 1)]);
        float c_n2 = std::max(eps, cfa[(r - 2) * w + c]);
        float c_s2 = std::max(eps, cfa[(r + 2) * w + c]);
        float c_w2 = std::max(eps, cfa[r * w + (c - 2)]);
        float c_e2 = std::max(eps, cfa[r * w + (c + 2)]);
        float c_n3 = std::max(eps, cfa[(r - 3) * w + c]);
        float c_s3 = std::max(eps, cfa[(r + 3) * w + c]);
        float c_w3 = std::max(eps, cfa[r * w + (c - 3)]);
        float c_e3 = std::max(eps, cfa[r * w + (c + 3)]);
        float c_n4 = std::max(eps, cfa[(r - 4) * w + c]);
        float c_s4 = std::max(eps, cfa[(r + 4) * w + c]);
        float c_w4 = std::max(eps, cfa[r * w + (c - 4)]);
        float c_e4 = std::max(eps, cfa[r * w + (c + 4)]);

        // Vertical and horizontal directional activity statistics
        float v_stat = std::max(
            epssq,
            -18.f * c0 * (c_n1 + c_s1) - 38.f * c0 * (c_n2 + c_s2) + 6.f * c0 * (c_n3 + c_s3) -
                2.f * c0 * (c_n4 + c_s4) + 38.f * c0 * c0 + 19.f * (c_n1 * c_n1 + c_s1 * c_s1) +
                12.f * (c_n2 * c_n2 + c_s2 * c_s2) + 3.f * (c_n3 * c_n3 + c_s3 * c_s3) +
                0.5f * (c_n4 * c_n4 + c_s4 * c_s4) + 14.f * c_n1 * c_s1 + 6.f * c_n2 * c_s2 -
                6.f * (c_n1 * c_n3 + c_s1 * c_s3) + 2.f * (c_n1 * c_s2 + c_s1 * c_n2) +
                10.f * (c_n1 * c_n2 + c_s1 * c_s2) - 2.f * (c_n2 * c_n3 + c_s2 * c_s3) +
                2.f * (c_n2 * c_n4 + c_s2 * c_s4)
        );
        float h_stat = std::max(
            epssq,
            -18.f * c0 * (c_w1 + c_e1) - 38.f * c0 * (c_w2 + c_e2) + 6.f * c0 * (c_w3 + c_e3) -
                2.f * c0 * (c_w4 + c_e4) + 38.f * c0 * c0 + 19.f * (c_w1 * c_w1 + c_e1 * c_e1) +
                12.f * (c_w2 * c_w2 + c_e2 * c_e2) + 3.f * (c_w3 * c_w3 + c_e3 * c_e3) +
                0.5f * (c_w4 * c_w4 + c_e4 * c_e4) + 14.f * c_w1 * c_e1 + 6.f * c_w2 * c_e2 -
                6.f * (c_w1 * c_w3 + c_e1 * c_e3) + 2.f * (c_w1 * c_e2 + c_e1 * c_w2) +
                10.f * (c_w1 * c_w2 + c_e1 * c_e2) - 2.f * (c_w2 * c_w3 + c_e2 * c_e3) +
                2.f * (c_w2 * c_w4 + c_e2 * c_e4)
        );

        // Low-pass filtered ratio + second-order Laplacian directional green predictors
        float lpf_c = 0.5f * c0 + 0.25f * (c_n1 + c_s1);
        float lpf_n = 0.5f * (c_n1 + c_n3) + 0.25f * (c0 + 2.f * c_n2 + c_n4);
        float lpf_s = 0.5f * (c_s1 + c_s3) + 0.25f * (c0 + 2.f * c_s2 + c_s4);
        float pv_n = c_n1 * lpf_c / std::max(eps, 0.5f * (lpf_c + lpf_n));
        float pv_s = c_s1 * lpf_c / std::max(eps, 0.5f * (lpf_c + lpf_s));
        float gv_ratio = 0.5f * (pv_n + pv_s);
        float gv_diff = 0.5f * (c_n1 + c_s1) + 0.25f * (2.f * c0 - c_n2 - c_s2);
        float gv = 0.5f * (gv_ratio + gv_diff);

        float lpf_ch = 0.5f * c0 + 0.25f * (c_w1 + c_e1);
        float lpf_w = 0.5f * (c_w1 + c_w3) + 0.25f * (c0 + 2.f * c_w2 + c_w4);
        float lpf_e = 0.5f * (c_e1 + c_e3) + 0.25f * (c0 + 2.f * c_e2 + c_e4);
        float ph_w = c_w1 * lpf_ch / std::max(eps, 0.5f * (lpf_ch + lpf_w));
        float ph_e = c_e1 * lpf_ch / std::max(eps, 0.5f * (lpf_ch + lpf_e));
        float gh_ratio = 0.5f * (ph_w + ph_e);
        float gh_diff = 0.5f * (c_w1 + c_e1) + 0.25f * (2.f * c0 - c_w2 - c_e2);
        float gh = 0.5f * (gh_ratio + gh_diff);

        float vh_disc = h_stat / (v_stat + h_stat);
        float g_est = vh_disc * gv + (1.f - vh_disc) * gh;
        float g_min = std::min({c_n1, c_s1, c_w1, c_e1});
        float g_max = std::max({c_n1, c_s1, c_w1, c_e1});
        out_rgb[idx * 3 + 1] = std::max(0.f, clampf(g_est, g_min * 0.85f, g_max * 1.15f));
      }
    }
  });

  complete_rb_from_green(cfa, w, h, pat, out_rgb);
}

// ============================================================================
// 2. LMMSE (Zhang & Wu 2005 Directional Linear Minimum Mean Square Error)
// ============================================================================
void ol_demosaic_bayer_lmmse(
    const float *cfa,
    size_t w,
    size_t h,
    int cfa_pattern,
    float *out_rgb
) {
  int pat[2][2];
  decode_cfa_pat(cfa_pattern, pat);
  fill_bayer_border(cfa, w, h, pat, 4, out_rgb);
  if (w < 9 || h < 9) return;

  // Allocate directional primary difference buffers: dH = G_H - C, dV = G_V - C
  std::vector<float> dH(w * h, 0.f);
  std::vector<float> dV(w * h, 0.f);

  // Step 1: Second-order Hamilton-Adams directional difference signals everywhere
  parallel_rows(2, h - 2, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r) {
      for (size_t c = 2; c + 2 < w; ++c) {
        size_t idx = r * w + c;
        int col = pat[r & 1][c & 1];
        float c0 = cfa[idx];
        float h1 = 0.5f * (cfa[r * w + c - 1] + cfa[r * w + c + 1]) +
                   0.25f * (2.f * c0 - cfa[r * w + c - 2] - cfa[r * w + c + 2]);
        float v1 = 0.5f * (cfa[(r - 1) * w + c] + cfa[(r + 1) * w + c]) +
                   0.25f * (2.f * c0 - cfa[(r - 2) * w + c] - cfa[(r + 2) * w + c]);
        if (col == 1) {
          // At Green sites, d = G - C_interp
          dH[idx] = c0 - h1;
          dV[idx] = c0 - v1;
          out_rgb[idx * 3 + 1] = c0;
        } else {
          // At R/B sites, d = G_interp - C
          dH[idx] = h1 - c0;
          dV[idx] = v1 - c0;
        }
      }
    }
  });

  // Step 2: 9-tap Gaussian low-pass filter and MMSE variance estimation at R/B sites
  static const float kGauss[9] = {
      0.023285f, 0.056662f, 0.121009f, 0.192065f, 0.213958f,
      0.192065f, 0.121009f, 0.056662f, 0.023285f
  };
  const float eps = 1e-7f;

  parallel_rows(4, h - 4, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r) {
      for (size_t c = 4; c + 4 < w; ++c) {
        size_t idx = r * w + c;
        int col = pat[r & 1][c & 1];
        if (col == 1) continue;

        // Smoothed horizontal and vertical difference signals in a 9-pixel neighborhood
        float fh[9], fv[9], rh[9], rv[9];
        float mu_h = 0.f, mu_v = 0.f;
        for (int k = -4; k <= 4; ++k) {
          size_t ih = r * w + (size_t)((ptrdiff_t)c + k);
          size_t iv = (size_t)((ptrdiff_t)r + k) * w + c;
          rh[k + 4] = dH[ih];
          rv[k + 4] = dV[iv];
          mu_h += kGauss[k + 4] * rh[k + 4];
          mu_v += kGauss[k + 4] * rv[k + 4];
        }
        // Estimate local signal variance and interpolation noise variance along H and V
        float var_sig_h = 0.f, var_err_h = 0.f;
        float var_sig_v = 0.f, var_err_v = 0.f;
        for (int k = 0; k < 9; ++k) {
          float dh_s = rh[k] - mu_h;
          float dv_s = rv[k] - mu_v;
          var_sig_h += kGauss[k] * dh_s * dh_s;
          var_sig_v += kGauss[k] * dv_s * dv_s;
          // High-frequency residual between noisy and locally smoothed difference
          fh[k] = 0.5f * (rh[k] + mu_h);
          fv[k] = 0.5f * (rv[k] + mu_v);
          float eh = rh[k] - fh[k];
          float ev = rv[k] - fv[k];
          var_err_h += kGauss[k] * eh * eh;
          var_err_v += kGauss[k] * ev * ev;
        }
        // LMMSE optimal scalar estimate along H and V
        float gain_h = var_sig_h / (var_sig_h + var_err_h + eps);
        float gain_v = var_sig_v / (var_sig_v + var_err_v + eps);
        float est_h = mu_h + gain_h * (rh[4] - mu_h);
        float est_v = mu_v + gain_v * (rv[4] - mu_v);

        // Posterior error variance for MMSE fusion weights
        float post_h = (1.f - gain_h) * var_sig_h + eps;
        float post_v = (1.f - gain_v) * var_sig_v + eps;
        float wh = 1.f / post_h;
        float wv = 1.f / post_v;
        float d_opt = (wh * est_h + wv * est_v) / (wh + wv);
        out_rgb[idx * 3 + 1] = std::max(0.f, cfa[idx] + d_opt);
      }
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
  int pat[2][2];
  decode_cfa_pat(cfa_pattern, pat);
  fill_bayer_border(cfa, w, h, pat, 4, out_rgb);
  if (w < 9 || h < 9) return;

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
// 4. Fujifilm 6x6 X-Trans Markesteijn-Style Directional Demosaicing
// ============================================================================
void ol_demosaic_xtrans_markesteijn(
    const float *cfa,
    size_t w,
    size_t h,
    const uint8_t xtrans_pat[36],
    float *out_rgb
) {
  const float eps = 1e-5f;
  // Pass 1: Reconstruct Green (channel 1) at non-Green sites using directional 4-neighbor weights
  parallel_rows(0, h, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r) {
      for (size_t c = 0; c < w; ++c) {
        size_t idx = r * w + c;
        int col = xtrans_pat[(r % 6) * 6 + (c % 6)] % 3;
        out_rgb[idx * 3 + 0] = 0.f;
        out_rgb[idx * 3 + 1] = 0.f;
        out_rgb[idx * 3 + 2] = 0.f;
        out_rgb[idx * 3 + col] = cfa[idx];
        if (col == 1) continue;

        float g_sum = 0.f, w_sum = 0.f;
        for (int dr = -2; dr <= 2; ++dr) {
          for (int dc = -2; dc <= 2; ++dc) {
            if (dr == 0 && dc == 0) continue;
            ptrdiff_t nr = std::clamp<ptrdiff_t>((ptrdiff_t)r + dr, 0, (ptrdiff_t)h - 1);
            ptrdiff_t nc = std::clamp<ptrdiff_t>((ptrdiff_t)c + dc, 0, (ptrdiff_t)w - 1);
            int ncol = xtrans_pat[(nr % 6) * 6 + (nc % 6)] % 3;
            if (ncol != 1) continue;
            float gv = cfa[(size_t)nr * w + (size_t)nc];
            ptrdiff_t orow = std::clamp<ptrdiff_t>((ptrdiff_t)r - dr, 0, (ptrdiff_t)h - 1);
            ptrdiff_t ocol = std::clamp<ptrdiff_t>((ptrdiff_t)c - dc, 0, (ptrdiff_t)w - 1);
            float grad = std::fabs(gv - cfa[(size_t)orow * w + (size_t)ocol]) + eps;
            float dist = (float)(dr * dr + dc * dc);
            float wt = 1.f / (dist * (grad + 0.01f));
            g_sum += wt * gv;
            w_sum += wt;
          }
        }
        out_rgb[idx * 3 + 1] = w_sum > 0.f ? g_sum / w_sum : cfa[idx];
      }
    }
  });

  // Pass 2: Interpolate (R - G) and (B - G) over the 5x5 X-Trans neighborhood
  parallel_rows(0, h, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r) {
      for (size_t c = 0; c < w; ++c) {
        size_t idx = r * w + c;
        int col = xtrans_pat[(r % 6) * 6 + (c % 6)] % 3;
        float g0 = out_rgb[idx * 3 + 1];
        for (int ch = 0; ch <= 2; ch += 2) {
          if (col == ch) continue;
          float d_sum = 0.f, w_sum = 0.f;
          for (int dr = -2; dr <= 2; ++dr) {
            for (int dc = -2; dc <= 2; ++dc) {
              ptrdiff_t nr = std::clamp<ptrdiff_t>((ptrdiff_t)r + dr, 0, (ptrdiff_t)h - 1);
              ptrdiff_t nc = std::clamp<ptrdiff_t>((ptrdiff_t)c + dc, 0, (ptrdiff_t)w - 1);
              int ncol = xtrans_pat[(nr % 6) * 6 + (nc % 6)] % 3;
              if (ncol != ch) continue;
              size_t nidx = (size_t)nr * w + (size_t)nc;
              float diff = cfa[nidx] - out_rgb[nidx * 3 + 1];
              float dist = (float)(dr * dr + dc * dc);
              float g_grad = std::fabs(g0 - out_rgb[nidx * 3 + 1]) + 0.02f;
              float wt = 1.f / (dist * g_grad);
              d_sum += wt * diff;
              w_sum += wt;
            }
          }
          out_rgb[idx * 3 + ch] = std::max(0.f, g0 + (w_sum > 0.f ? d_sum / w_sum : 0.f));
        }
      }
    }
  });
}

// ============================================================================
// 5. Optical Black-Point Calibration, Highlight Recovery & Sensor Denoising
// ============================================================================
void ol_raw_calibrate_and_reconstruct(
    float *rgb_interleaved,
    size_t w,
    size_t h,
    const float black_levels[4],
    float white_level,
    int highlight_mode
) {
  float b_r = black_levels ? black_levels[0] : 0.f;
  float b_g = black_levels ? 0.5f * (black_levels[1] + black_levels[3]) : 0.f;
  float b_b = black_levels ? black_levels[2] : 0.f;
  float b[3] = {b_r, b_g, b_b};
  float wl = white_level > 1e-4f ? white_level : 1.f;

  parallel_rows(0, h, [&](size_t r0, size_t r1) {
    for (size_t r = r0; r < r1; ++r) {
      for (size_t c = 0; c < w; ++c) {
        float *p = rgb_interleaved + (r * w + c) * 3;
        for (int k = 0; k < 3; ++k) {
          float denom = std::max(1e-5f, wl - b[k]);
          p[k] = std::max(0.f, (p[k] - b[k]) / denom);
        }
        if (highlight_mode == 0) {
          p[0] = std::min(1.f, p[0]);
          p[1] = std::min(1.f, p[1]);
          p[2] = std::min(1.f, p[2]);
        } else if (highlight_mode == 2) {
          // Blend: smoothly pull clipped channels toward neutral highlight luminance
          float mx = std::max({p[0], p[1], p[2]});
          if (mx > 1.f) {
            float mn = std::min({p[0], p[1], p[2]});
            float luma = 0.2627f * p[0] + 0.6780f * p[1] + 0.0593f * p[2];
            float t = clampf((mx - 1.f) / std::max(0.15f, mx - mn), 0.f, 1.f);
            p[0] = p[0] * (1.f - t) + luma * t;
            p[1] = p[1] * (1.f - t) + luma * t;
            p[2] = p[2] * (1.f - t) + luma * t;
          }
        } else if (highlight_mode >= 3) {
          // Rebuild: reconstruct blown channel ratios from unclipped channels
          int clipped = (p[0] >= 0.99f) + (p[1] >= 0.99f) + (p[2] >= 0.99f);
          if (clipped == 1 || clipped == 2) {
            float unclipped_sum = 0.f;
            int unclipped_cnt = 0;
            for (int k = 0; k < 3; ++k) {
              if (p[k] < 0.99f) {
                unclipped_sum += p[k];
                unclipped_cnt++;
              }
            }
            float ref = unclipped_cnt ? (unclipped_sum / (float)unclipped_cnt) : 1.f;
            for (int k = 0; k < 3; ++k) {
              if (p[k] >= 0.99f) {
                p[k] = std::max(p[k], ref * 1.15f);
              }
            }
          }
        }
      }
    }
  });
}

void ol_raw_refilter_demosaic(
    const float *in_rgb,
    size_t w,
    size_t h,
    int algo,
    float *out_rgb
) {
  if (w < 9 || h < 9 || algo <= 0) {
    std::memcpy(out_rgb, in_rgb, w * h * 3 * sizeof(float));
    return;
  }
  std::vector<float> cfa(w * h);
  if (algo == 4) {
    static const uint8_t kXtrans[36] = {
        1, 1, 0, 1, 1, 2,
        1, 1, 2, 1, 1, 0,
        2, 0, 1, 0, 2, 1,
        1, 1, 2, 1, 1, 0,
        1, 1, 0, 1, 1, 2,
        0, 2, 1, 2, 0, 1
    };
    for (size_t r = 0; r < h; ++r)
      for (size_t c = 0; c < w; ++c)
        cfa[r * w + c] = in_rgb[(r * w + c) * 3 + kXtrans[(r % 6) * 6 + (c % 6)]];
    ol_demosaic_xtrans_markesteijn(cfa.data(), w, h, kXtrans, out_rgb);
  } else {
    int pat[2][2] = {{0, 1}, {1, 2}};
    for (size_t r = 0; r < h; ++r)
      for (size_t c = 0; c < w; ++c)
        cfa[r * w + c] = in_rgb[(r * w + c) * 3 + pat[r & 1][c & 1]];
    if (algo == 1) ol_demosaic_bayer_rcd(cfa.data(), w, h, 0, out_rgb);
    else if (algo == 2) ol_demosaic_bayer_amaze(cfa.data(), w, h, 0, out_rgb);
    else ol_demosaic_bayer_lmmse(cfa.data(), w, h, 0, out_rgb);
  }
}

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
// 6. Real-Time RGB Parade & Luma/Chroma Waveform Scopes (C++ Fallback / Native)
// ============================================================================
void ol_compute_scopes_parade_waveform(
    const uint8_t *rgba,
    size_t w,
    size_t h,
    size_t scope_w,
    size_t scope_h,
    uint8_t *out_parade_rgba,
    uint8_t *out_waveform_rgba
) {
  if (!rgba || w == 0 || h == 0 || scope_w < 3 || scope_h < 2) return;
  std::vector<uint32_t> p_hist(scope_w * scope_h * 3, 0);
  std::vector<uint32_t> w_hist(scope_w * scope_h * 3, 0);

  size_t third = scope_w / 3;
  for (size_t y = 0; y < h; ++y) {
    for (size_t x = 0; x < w; ++x) {
      const uint8_t *px = rgba + (y * w + x) * 4;
      size_t sx = (x * scope_w) / w;
      size_t px_third = (x * third) / w;
      for (int c = 0; c < 3; ++c) {
        size_t sy = scope_h - 1 - ((size_t)px[c] * (scope_h - 1)) / 255;
        size_t p_x = std::min(scope_w - 1, (size_t)c * third + px_third);
        p_hist[(sy * scope_w + p_x) * 3 + (size_t)c]++;
        w_hist[(sy * scope_w + sx) * 3 + (size_t)c]++;
      }
    }
  }

  float norm = 3.5f / std::max(1.f, (float)h / (float)scope_h);
  for (size_t i = 0; i < scope_w * scope_h; ++i) {
    if (out_parade_rgba) {
      for (int c = 0; c < 3; ++c) {
        float v = 1.f - std::exp(-(float)p_hist[i * 3 + (size_t)c] * norm);
        out_parade_rgba[i * 4 + (size_t)c] = (uint8_t)clampf(v * 255.f + 10.f, 0.f, 255.f);
      }
      out_parade_rgba[i * 4 + 3] = 255;
    }
    if (out_waveform_rgba) {
      for (int c = 0; c < 3; ++c) {
        float v = 1.f - std::exp(-(float)w_hist[i * 3 + (size_t)c] * norm);
        out_waveform_rgba[i * 4 + (size_t)c] = (uint8_t)clampf(v * 255.f + 10.f, 0.f, 255.f);
      }
      out_waveform_rgba[i * 4 + 3] = 255;
    }
  }
}

// ============================================================================
// 7. Focus-Peaking, Blink Detection & 16-Bit Guided Alpha Mask Feathering
// ============================================================================
void ol_compute_focus_peaking_and_blink(
    const float *rgb,
    size_t w,
    size_t h,
    float threshold,
    const float *face_boxes_xywh,
    size_t num_faces,
    uint8_t *out_peaking_mask,
    float *out_focus_score,
    float *out_eye_openness,
    int *out_blink_detected
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
  float ear_min = 0.32f;
  int blink = 0;

  if (face_boxes_xywh && num_faces > 0) {
    double eye_sharp_sum = 0.0;
    size_t eye_sharp_cnt = 0;
    for (size_t f = 0; f < num_faces; ++f) {
      float fx = clampf(face_boxes_xywh[f * 4 + 0], 0.f, 1.f);
      float fy = clampf(face_boxes_xywh[f * 4 + 1], 0.f, 1.f);
      float fw = clampf(face_boxes_xywh[f * 4 + 2], 0.01f, 1.f - fx);
      float fh = clampf(face_boxes_xywh[f * 4 + 3], 0.01f, 1.f - fy);
      // Upper eye band: 22%..48% of face height, 15%..85% of face width
      size_t r0 = std::max<size_t>(1, (size_t)((fy + 0.22f * fh) * (float)h));
      size_t r1 = std::min<size_t>(h - 1, (size_t)((fy + 0.48f * fh) * (float)h));
      size_t c0 = std::max<size_t>(1, (size_t)((fx + 0.15f * fw) * (float)w));
      size_t c1 = std::min<size_t>(w - 1, (size_t)((fx + 0.85f * fw) * (float)w));
      double v_grad = 0.0, h_grad = 0.0;
      for (size_t r = r0; r < r1; ++r) {
        for (size_t c = c0; c < c1; ++c) {
          size_t idx = r * w + c;
          float gx = std::fabs(luma[idx + 1] - luma[idx - 1]);
          float gy = std::fabs(luma[idx + w] - luma[idx - w]);
          h_grad += gx;
          v_grad += gy;
          eye_sharp_sum += (double)(gx * gx + gy * gy);
          eye_sharp_cnt++;
        }
      }
      // Open eyes show strong horizontal sclera/iris contrast relative to vertical eyelid line
      float ear = (float)(0.14 + 0.26 * (h_grad / std::max(1e-5, v_grad + h_grad)));
      ear_min = std::min(ear_min, ear);
      if (ear < 0.205f) blink = 1;
    }
    if (eye_sharp_cnt > 0) {
      global_score = (float)std::min(100.0, std::sqrt(eye_sharp_sum / (double)eye_sharp_cnt) * 650.0);
    }
  }
  if (out_focus_score) *out_focus_score = global_score;
  if (out_eye_openness) *out_eye_openness = ear_min;
  if (out_blink_detected) *out_blink_detected = blink;
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
    raw->ahd_interpolate();
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

