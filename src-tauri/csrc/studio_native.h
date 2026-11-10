#ifndef OPTIXEL_STUDIO_NATIVE_H
#define OPTIXEL_STUDIO_NATIVE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// A. Camera RAW Engine, Multi-Algorithm Demosaicing & Optical Calibration
// ============================================================================

/// Demosaicing algorithm IDs:
/// 0 = Bilinear / Fast
/// 1 = RCD (Ratio Corrected Demosaicing — general sharpness)
/// 2 = AMaZE (Aliasing Minimization and Zipper Elimination — low-ISO detail)
/// 3 = LMMSE (Linear Minimum Mean Square Error — high-ISO / astro / night)
/// 4 = X-Trans Markesteijn (Fujifilm 6x6 non-Bayer sensor array)

/// Highlight reconstruction modes:
/// 0 = Clip (standard [0, 1] clamp)
/// 1 = Unclip (preserve extended linear float highlights > 1.0)
/// 2 = Blend (blend partially clipped channels toward neutral luma ratio)
/// 3 = Rebuild (LCh chromaticity & spatial gradient propagation into blown channels)

/// Per-channel optical black-point calibration and multi-channel highlight recovery.
void ol_raw_calibrate_and_reconstruct(
    float* rgb_interleaved,
    size_t w,
    size_t h,
    const float black_levels[4],
    float white_level,
    int highlight_mode
);

/// RCD (Ratio Corrected Demosaicing) on a single-channel Bayer CFA plane -> interleaved RGB.
/// `cfa_pattern`: 0 = RGGB, 1 = BGGR, 2 = GRBG, 3 = GBRG.
void ol_demosaic_bayer_rcd(
    const float* cfa,
    size_t w,
    size_t h,
    int cfa_pattern,
    float* out_rgb
);

/// AMaZE (Aliasing Minimization and Zipper Elimination) on a Bayer CFA plane -> interleaved RGB.
void ol_demosaic_bayer_amaze(
    const float* cfa,
    size_t w,
    size_t h,
    int cfa_pattern,
    float* out_rgb
);

/// LMMSE (Linear Minimum Mean Square Error) on a Bayer CFA plane -> interleaved RGB.
void ol_demosaic_bayer_lmmse(
    const float* cfa,
    size_t w,
    size_t h,
    int cfa_pattern,
    float* out_rgb
);

/// Markesteijn-style directional demosaicing on a Fujifilm 6x6 X-Trans CFA plane -> interleaved RGB.
/// `xtrans_pat`: 36-element array (6x6) with values 0=R, 1=G, 2=B.
void ol_demosaic_xtrans_markesteijn(
    const float* cfa,
    size_t w,
    size_t h,
    const uint8_t xtrans_pat[36],
    float* out_rgb
);

/// Re-mosaic & apply selected demosaic algorithm (`1=RCD`, `2=AMaZE`, `3=LMMSE`, `4=X-Trans`)
/// to an already-decoded linear RGB buffer to refine directional micro-detail or suppress zipper/chroma noise.
void ol_raw_refilter_demosaic(
    const float* in_rgb,
    size_t w,
    size_t h,
    int algo,
    float* out_rgb
);

/// Pre-color-conversion linear sensor denoising (Anscombe-stabilized bilateral + hot-pixel impulse suppression).
void ol_raw_sensor_denoise_linear(
    float* rgb_linear,
    size_t w,
    size_t h,
    float strength
);

// ============================================================================
// C. MakerNotes Binary Parser (Canon / Nikon / Sony / Fujifilm)
// ============================================================================

#define OL_MAX_AF_POINTS 64

typedef struct {
    float x;       // normalized center X in [0, 1]
    float y;       // normalized center Y in [0, 1]
    float w;       // normalized width in [0, 1]
    float h;       // normalized height in [0, 1]
    uint8_t active; // 1 = active / in-focus AF point, 0 = inactive AF point
    uint8_t kind;   // 0 = phase/single, 1 = eye-AF, 2 = face-AF, 3 = zone/tracking
} OlAfPoint;

typedef struct {
    uint32_t shutter_count;
    uint32_t af_count;
    OlAfPoint af_points[OL_MAX_AF_POINTS];
    float focus_distance_m;
    uint8_t is_xtrans;
    uint8_t vendor_id; // 0=Unknown, 1=Canon, 2=Nikon, 3=Sony, 4=Fujifilm, 5=Olympus, 6=Panasonic, 7=Pentax
    char lens_serial[64];
    char body_serial[64];
    char camera_model[64];
    char cfa_pattern[16];
    uint32_t black_levels[4];
    uint32_t border_black_counts[4];
    uint32_t white_level;
    uint32_t linear_max[4];
    float cam_mul[4];
    uint32_t raw_width;
    uint32_t raw_height;
} OlMakerNotesOut;

/// Parse binary TIFF/EXIF/CR3/MakerNotes payload to extract shutter count, lens serial, and AF points.
int ol_parse_makernotes(
    const uint8_t* data,
    size_t len,
    OlMakerNotesOut* out
);

/// Lossless DCT JPEG transform via libturbojpeg (`op`: 1=HFlip, 2=VFlip, 5=Rot90, 6=Rot180, 7=Rot270;
/// optional crop when `crop_w > 0 && crop_h > 0`). Caller frees `*out_jpeg` with `ol_free_buf`.
int ol_jpeg_lossless_transform(
    const uint8_t* jpeg_in,
    size_t len_in,
    int op,
    int crop_x,
    int crop_y,
    int crop_w,
    int crop_h,
    uint8_t** out_jpeg,
    size_t* out_len,
    char err_buf[160]
);

/// Decode HEIC / HEIF / 10-bit AVIF container into oriented 8-bit RGB + optional embedded ICC profile.
/// Caller frees `*out_rgb` and `*out_icc` with `ol_free_buf`.
int ol_decode_heif_container(
    const uint8_t* data,
    size_t len,
    int max_edge,
    uint8_t** out_rgb,
    uint32_t* out_w,
    uint32_t* out_h,
    uint8_t** out_icc,
    size_t* out_icc_len,
    char err_buf[160]
);

void ol_free_buf(void* p);

// ============================================================================
// D. Real-Time RGB Parade & Luma/Chroma Waveform Scopes
// ============================================================================

/// Accumulate and render an RGBA8 image into an RGB Parade buffer and a Luma/Chroma Waveform buffer
/// (`scope_w` x `scope_h` x 4 bytes each).
void ol_compute_scopes_parade_waveform(
    const uint8_t* rgba,
    size_t w,
    size_t h,
    size_t scope_w,
    size_t scope_h,
    uint8_t* out_parade_rgba,
    uint8_t* out_waveform_rgba
);

// ============================================================================
// E. Focus-Peaking, Blink Detection & 16-Bit Guided Alpha Mask Feathering
// ============================================================================

/// Compute per-pixel focus-peaking mask (`0..255`), global/subject focus score (`0..100`),
/// and eye openness aspect ratio (EAR) over optional normalized face boxes `[x, y, w, h]`.
void ol_compute_focus_peaking_and_blink(
    const float* rgb,
    size_t w,
    size_t h,
    float threshold,
    const float* face_boxes_xywh,
    size_t num_faces,
    uint8_t* out_peaking_mask,
    float* out_focus_score,
    float* out_eye_openness,
    int* out_blink_detected
);

/// O(N) He et al. Guided Filter producing 16-bit (`0..65535`) and float (`0..1`) feathered alpha masks.
void ol_guided_feather_mask_16bit(
    const float* guide_luma,
    const float* raw_alpha,
    size_t w,
    size_t h,
    int radius,
    float eps,
    uint16_t* out_alpha16,
    float* out_alpha_f32
);

#ifdef __cplusplus
}
#endif

#endif // OPTIXEL_STUDIO_NATIVE_H

