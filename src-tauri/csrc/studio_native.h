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

/// Linear sensor denoising (Anscombe-stabilized bilateral + hot-pixel impulse suppression), run by
/// ol_raw.cpp on the demosaiced camera RGB just before LibRaw's colour matrix.
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

/// Decode HEIC / HEIF / 10-bit AVIF container into oriented 16-bit RGB + optional embedded ICC profile.
/// Caller frees `*out_rgb` and `*out_icc` with `ol_free_buf`.
int ol_decode_heif_container(
    const uint8_t* data,
    size_t len,
    int max_edge,
    uint16_t** out_rgb,
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

/// RGB parade bins (`scope_w` x `scope_h` x 3: each channel in its third) and luma waveform bins
/// (`scope_w` x `scope_h` x 4: red, green, blue sums at 16 per full channel plus 1, and the count) of an
/// RGBA8 image, exactly as shaders/scopes.wgsl accumulates them. Both start zeroed.
void ol_scope_bins(
    const uint8_t* rgba,
    size_t w,
    size_t h,
    size_t scope_w,
    size_t scope_h,
    uint32_t* parade,
    uint32_t* waveform
);

// ============================================================================
// E. Focus peaking, eye focus & 16-bit guided alpha mask feathering
// ============================================================================

/// Per-pixel focus-peaking mask (`0..255`) and a focus score (`0..100`): of the eyes when there are faces
/// (`eyes_xywh`: per face, its two eye patches, 8 floats: x, y, w, h of each, normalized), else the frame.
/// (Closed eyes are judged from landmarks, in Rust: src/faces.rs.)
void ol_compute_focus_peaking(
    const float* rgb,
    size_t w,
    size_t h,
    float threshold,
    const float* eyes_xywh,
    size_t num_faces,
    uint8_t* out_peaking_mask,
    float* out_focus_score
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

