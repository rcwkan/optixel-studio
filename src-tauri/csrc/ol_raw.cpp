// Optixel Studio's bridge into LibRaw (vendor/libraw) + MakerNotes & optical calibration reader.
// Rust side: src/libraw.rs. No C++ exception may leave this file.

#include "ol_raw.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>

namespace {

inline uint16_t rd16(const uint8_t *p, bool le) {
  return le ? (uint16_t)(p[0] | (p[1] << 8)) : (uint16_t)((p[0] << 8) | p[1]);
}

inline int16_t rdi16(const uint8_t *p, bool le) {
  return (int16_t)rd16(p, le);
}

inline uint32_t rd32(const uint8_t *p, bool le) {
  return le ? (uint32_t)(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24))
            : (uint32_t)((p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]);
}

inline float clamp01(float x) {
  return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}

inline bool ieq_prefix(const char *s, const char *prefix) {
  while (*prefix) {
    char a = *s++;
    char b = *prefix++;
    if (a >= 'A' && a <= 'Z') a += 'a' - 'A';
    if (b >= 'A' && b <= 'Z') b += 'a' - 'A';
    if (a != b) return false;
  }
  return true;
}

void push_af(OlMakerNotesOut *out, float x, float y, float w, float h, uint8_t active, uint8_t kind) {
  if (out->af_count >= OL_MAX_AF_POINTS) return;
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(w) || !std::isfinite(h)) return;
  w = std::clamp(w, 0.015f, 0.95f);
  h = std::clamp(h, 0.015f, 0.95f);
  x = std::clamp(x, w * 0.5f, 1.f - w * 0.5f);
  y = std::clamp(y, h * 0.5f, 1.f - h * 0.5f);
  OlAfPoint &pt = out->af_points[out->af_count++];
  pt.x = x;
  pt.y = y;
  pt.w = w;
  pt.h = h;
  pt.active = active ? 1 : 0;
  pt.kind = kind;
}

// Walk TIFF / EXIF / Nikon MakerNote IFDs in `data[0..len]` for the Nikon MakerNote's 0x00a7 ShutterCount
// (LibRaw reads it as a decryption key without keeping it). Only inside a Nikon MakerNote: other
// vendors use 0x00a7 for other things.
void scan_tiff_shutter_count(const uint8_t *tiff, size_t len, uint32_t *shutter_out) {
  if (len < 16 || *shutter_out > 0) return;
  bool le = false;
  if (tiff[0] == 'I' && tiff[1] == 'I' && tiff[2] == 42 && tiff[3] == 0) le = true;
  else if (tiff[0] == 'M' && tiff[1] == 'M' && tiff[2] == 0 && tiff[3] == 42) le = false;
  else return;

  auto walk_ifd = [&](auto &self, size_t off, size_t base, bool endian, int depth, bool nikon) -> void {
    if (depth > 5 || off + 2 > len || *shutter_out > 0) return;
    uint16_t n = rd16(tiff + off, endian);
    if (n > 512 || off + 2 + (size_t)n * 12 > len) return;
    for (uint16_t i = 0; i < n; ++i) {
      const uint8_t *e = tiff + off + 2 + (size_t)i * 12;
      uint16_t tag = rd16(e + 0, endian);
      uint16_t typ = rd16(e + 2, endian);
      uint32_t cnt = rd32(e + 4, endian);
      uint32_t val = rd32(e + 8, endian);
      if (nikon && tag == 0x00a7 && typ == 4 && cnt == 1 && val > 0 && val < 5000000) {
        *shutter_out = val;
        return;
      }
      // SubIFDs (0x014a), ExifIFD (0x8769)
      if (!nikon && (tag == 0x8769 || tag == 0x014a) && (typ == 4 || typ == 13)) {
        if (cnt == 1) {
          self(self, base + val, base, endian, depth + 1, false);
        } else if (base + val + (size_t)cnt * 4 <= len && cnt <= 8) {
          for (uint32_t k = 0; k < cnt; ++k) {
            uint32_t sub = rd32(tiff + base + val + k * 4, endian);
            self(self, base + sub, base, endian, depth + 1, false);
          }
        }
      }
      // Nikon MakerNote (0x927c): "Nikon\0", 4 bytes, then a TIFF header its offsets are relative to
      if (!nikon && tag == 0x927c && cnt > 18 && base + val + cnt <= len) {
        const uint8_t *mn = tiff + base + val;
        if (std::memcmp(mn, "Nikon\0", 6) == 0 && (mn[10] == 'I' || mn[10] == 'M')) {
          bool mn_le = (mn[10] == 'I');
          uint32_t mn_ifd = rd32(mn + 14, mn_le);
          self(self, (base + val + 10) + mn_ifd, base + val + 10, mn_le, depth + 1, true);
        }
      }
    }
    if (depth == 0 && off + 2 + (size_t)n * 12 + 4 <= len) {
      uint32_t next_ifd = rd32(tiff + off + 2 + (size_t)n * 12, endian);
      if (next_ifd > 0 && next_ifd < len) {
        self(self, next_ifd, base, endian, depth + 1, false);
      }
    }
  };

  uint32_t ifd0 = rd32(tiff + 4, le);
  walk_ifd(walk_ifd, ifd0, 0, le, 0, false);
}

void scan_file_shutter_count(const uint8_t *data, size_t len, uint32_t *shutter_out) {
  if (!data || len < 16) return;
  // Direct TIFF / NEF / CR2 / ARW / DNG
  if ((data[0] == 'I' && data[1] == 'I') || (data[0] == 'M' && data[1] == 'M')) {
    scan_tiff_shutter_count(data, len, shutter_out);
    return;
  }
  // JPEG with APP1 Exif ("Exif\0\0")
  if (data[0] == 0xFF && data[1] == 0xD8) {
    size_t pos = 2;
    while (pos + 4 < len && data[pos] == 0xFF) {
      uint8_t marker = data[pos + 1];
      if (marker == 0xDA || marker == 0xD9) break;
      uint16_t seg_len = (uint16_t)((data[pos + 2] << 8) | data[pos + 3]);
      if (seg_len < 2 || pos + 2 + seg_len > len) break;
      if (marker == 0xE1 && seg_len > 14 && std::memcmp(data + pos + 4, "Exif\0\0", 6) == 0) {
        scan_tiff_shutter_count(data + pos + 10, seg_len - 8, shutter_out);
        break;
      }
      pos += 2 + seg_len;
    }
  }
}

// Decode Canon AFInfo2 (`0x0026` / `0x003c`) stored in `imCommon.afdata`
void decode_canon_af(const libraw_afinfo_item_t &af, OlMakerNotesOut *out) {
  const uint8_t *b = af.AFInfoData;
  unsigned len = af.AFInfoData_length;
  if (!b || len < 20) return;
  bool le = (af.AFInfoData_order == 0x4949);
  uint16_t af_mode = rd16(b + 2, le);
  uint16_t num_pts = rd16(b + 4, le);
  uint16_t valid_pts = rd16(b + 6, le);
  uint16_t af_img_w = rd16(b + 12, le);
  uint16_t af_img_h = rd16(b + 14, le);
  if (valid_pts == 0 || valid_pts > num_pts) valid_pts = num_pts;
  if (num_pts == 0 || num_pts > 1024 || af_img_w == 0 || af_img_h == 0) return;

  size_t need = 16 + (size_t)num_pts * 8;
  if (len < need) return;
  const uint8_t *widths = b + 16;
  const uint8_t *heights = widths + (size_t)num_pts * 2;
  const uint8_t *xpos = heights + (size_t)num_pts * 2;
  const uint8_t *ypos = xpos + (size_t)num_pts * 2;
  const uint8_t *in_focus = ypos + (size_t)num_pts * 2;
  size_t mask_words = (num_pts + 15) / 16;
  const uint8_t *selected = in_focus + mask_words * 2;
  bool has_mask = (len >= need + mask_words * 2);
  bool has_sel = (len >= need + mask_words * 4);

  // AFAreaMode: 5 Face Detect, 6 Face + Tracking; 4 Auto, 7 Zone, 11/12 Flexizone Multi, 14-22 Large /
  // Flexible Zone and Whole Area; the rest single points and their expansions
  uint8_t kind = (af_mode == 5 || af_mode == 6) ? 2
                 : (af_mode == 4 || af_mode == 7 || af_mode == 11 || af_mode == 12 || af_mode >= 14) ? 3 : 0;
  uint16_t limit = std::min<uint16_t>(valid_pts, OL_MAX_AF_POINTS);
  // First add active / in-focus points, then inactive points up to OL_MAX_AF_POINTS
  for (int pass = 0; pass < 2; ++pass) {
    for (uint16_t i = 0; i < valid_pts && out->af_count < OL_MAX_AF_POINTS; ++i) {
      bool act = false;
      if (has_mask) {
        uint16_t wmask = rd16(in_focus + (i / 16) * 2, le);
        act = ((wmask >> (i % 16)) & 1) != 0;
      }
      if (!act && has_sel) {
        uint16_t smask = rd16(selected + (i / 16) * 2, le);
        act = ((smask >> (i % 16)) & 1) != 0;
      }
      if ((pass == 0 && !act) || (pass == 1 && act)) continue;
      if (pass == 1 && i >= limit) break;
      uint16_t pw = rd16(widths + i * 2, le);
      uint16_t ph = rd16(heights + i * 2, le);
      int16_t px = rdi16(xpos + i * 2, le);
      int16_t py = rdi16(ypos + i * 2, le);
      float nx = 0.5f + (float)px / (float)af_img_w;
      float ny = 0.5f - (float)py / (float)af_img_h;
      float nw = pw > 0 ? (float)pw / (float)af_img_w : 0.04f;
      float nh = ph > 0 ? (float)ph / (float)af_img_h : 0.04f;
      push_af(out, nx, ny, nw, nh, act ? 1 : 0, kind);
    }
  }
}

// Nikon AFInfo2 (0x00b7), per ExifTool's AFInfo2V0100/0101/0300/0400 tables. LibRaw drops the 4-byte
// version string, so b[k] is ExifTool's offset k + 4.
// - With coordinates (contrast-detect / live view, and the mirrorless bodies): the AF area, its centre
//   and size in AFImageWidth x AFImageHeight.
// - Phase-detect, 51-point schema: every point, the used ones active. `crop` is the frame's crop factor
//   (1 FX, ~1.5 DX), as the module covers a fixed patch of the sensor.
void decode_nikon_af(const libraw_afinfo_item_t &af, float crop, OlMakerNotesOut *out) {
  const uint8_t *b = af.AFInfoData;
  unsigned len = af.AFInfoData_length;
  if (af.AFInfoData_tag != 0x00b7 || !b || len < 8) return;
  bool le = (af.AFInfoData_order == 0x4949);
  int ver = af.AFInfoData_version;

  // where AFImageWidth starts (then height, x, y, w, h, all int16u), whether the coordinates are filled in
  // (AFDetectionMethod not phase-detect, or AFCoordinatesAvailable), and the 51-point fields
  int box = -1, used = -1, primary = -1;
  bool coords = false;
  if (ver == 100) box = 12, coords = b[0] != 0, used = 4, primary = 3;
  else if (ver == 101) box = 66, coords = b[0] != 0, used = 4, primary = 64;
  else if (ver == 300) box = 38, coords = b[3] != 0, used = 6, primary = 52;
  else if (ver == 400) box = 58, coords = b[3] != 0;
  if (coords && box >= 0 && len >= (unsigned)box + 12) {
    float iw = rd16(b + box, le), ih = rd16(b + box + 2, le);
    float ax = rd16(b + box + 4, le), ay = rd16(b + box + 6, le);
    float aw = rd16(b + box + 8, le), ah = rd16(b + box + 10, le);
    if (iw > 0 && ih > 0 && ax <= iw && ay <= ih && aw > 0 && ah > 0) {
      push_af(out, ax / iw, ay / ih, aw / iw, ah / ih, 1, 0);
      return;
    }
  }
  // FocusPointSchema 1 = 51 points: rows A..E, columns 1..11 (A and E have 9: A1 sits in column 2).
  // Numbered (PrimaryAFPoint 1..51, AFPointsUsed bit n-1) column by column from the centre, each column's
  // rows in the order C, B, A, D, E. Spacing measured on a D800 frame (FX), as a fraction of the frame;
  // the three centre columns' rows are further apart.
  if (used < 0 || b[2] != 1 || len < (unsigned)used + 7 || len <= (unsigned)primary) return;
  static const int kCols[11] = {6, 7, 5, 8, 9, 10, 11, 4, 3, 2, 1};
  static const int kRows[5] = {2, 1, 0, 3, 4};
  static const float kColX[6] = {0.f, 0.0537f, 0.1080f, 0.1549f, 0.2018f, 0.2486f};
  float k = std::clamp(crop, 1.f, 2.f);
  int n = 0;
  for (int col : kCols) {
    for (int row : kRows) {
      if ((col == 1 || col == 11) && (row == 0 || row == 4)) continue;
      int c = col - 6, ac = c < 0 ? -c : c;
      float x = 0.5f + (c < 0 ? -1.f : 1.f) * kColX[ac] * k;
      float y = 0.5f + (row - 2) * (ac <= 1 ? 0.0753f : 0.0651f) * k;
      bool act = (b[used + n / 8] >> (n % 8)) & 1;
      ++n;
      act = act || n == b[primary];
      push_af(out, x, y, 0.041f * k, 0.049f * k, act ? 1 : 0, 0);
    }
  }
}

// Points from the sensor's landscape frame to the upright one the decoders hand out, by LibRaw's `flip`
// (dcraw: bit 4 swaps the axes of the output, then 2 mirrors source rows, 1 source columns).
void orient_af(OlMakerNotesOut *out, int flip) {
  for (uint32_t i = 0; i < out->af_count; ++i) {
    OlAfPoint &p = out->af_points[i];
    float x = (flip & 1) ? 1.f - p.x : p.x;
    float y = (flip & 2) ? 1.f - p.y : p.y;
    if (flip & 4) {
      p.x = y, p.y = x;
      std::swap(p.w, p.h);
    } else {
      p.x = x, p.y = y;
    }
  }
}

} // namespace

extern "C" {

typedef struct {
  int demosaic;     // LibRaw user_qual (0..12, -1 default) or OL_DEMOSAIC_RCD(101)/LMMSE(102)/AMAZE(103)
  int highlight;    // 0 clip, 1 unclip, 2 blend, 3-9 rebuild
  int proxy_edge;   // > 0: half-size fast binning when edge >= 2 * proxy_edge and demosaic < 0
  int black_offset; // ADU offset added to per-channel optical black level
  float raw_denoise; // 0..1 sensor denoise on camera RGB, before the colour matrix (OlRaw::denoise)
} OlRawOpts;

typedef struct {
  void *handle;              // free with ol_raw_free
  const unsigned short *px;  // width * height * colors samples, 65535 = white before `gain`
  int width, height, colors; // colors: 3, or 1 for a monochrome sensor
  float gain;                // px / 65535 * gain = linear light; > 1 when highlights were kept above white
  char error[160];
} OlRawOut;

static int fail(OlRawOut *out, const char *what, const char *why) {
  snprintf(out->error, sizeof out->error, "%s: %s", what, why);
  return 1;
}

int ol_raw_develop(const unsigned char *data, size_t len, const OlRawOpts *o, OlRawOut *out) {
  memset(out, 0, sizeof *out);
  libraw_processed_image_t *img = nullptr;
  try {
    std::unique_ptr<OlRaw> raw(new OlRaw); // too big for the stack of a worker thread
    libraw_output_params_t &p = raw->imgdata.params;
    p.output_color = 8; // Rec.2020, straight from the camera matrix
    p.output_bps = 16;
    p.gamm[0] = p.gamm[1] = 1.0; // linear
    p.no_auto_bright = 1;
    p.adjust_maximum_thr = 0; // white is the sensor's, not this frame's brightest pixel
    p.use_camera_wb = 1;
    if (o->demosaic >= 100) {
      raw->use_own_demosaic(o->demosaic);
      p.user_qual = 3; // fallback for X-Trans (3-pass Markesteijn)
    } else {
      raw->use_own_demosaic(0);
      p.user_qual = o->demosaic;
    }
    raw->use_denoise(o->raw_denoise);
    p.highlight = o->highlight;

    int rc = raw->open_buffer(data, len);
    if (rc != LIBRAW_SUCCESS) return fail(out, "open", LibRaw::strerror(rc));
    raw->adjust_to_raw_inset_crop(1); // the camera's own frame (the size its JPEGs have), when the file names one
    const libraw_image_sizes_t &s = raw->imgdata.sizes;
    int edge = s.width > s.height ? s.width : s.height;
    // Bin to half-size for fast proxy decodes unless a custom demosaic is being evaluated at full res
    p.half_size = o->proxy_edge > 0 && edge >= 2 * o->proxy_edge;
    if ((rc = raw->unpack()) != LIBRAW_SUCCESS) return fail(out, "unpack", LibRaw::strerror(rc));

    // Apply optional optical black-level offset in sensor ADU
    if (o->black_offset != 0) {
      int base = (int)raw->imgdata.color.black + o->black_offset;
      p.user_black = std::max(0, base);
    }

    if ((rc = raw->dcraw_process()) != LIBRAW_SUCCESS) return fail(out, "develop", LibRaw::strerror(rc));
    img = raw->dcraw_make_mem_image(&rc);
    if (!img) return fail(out, "image", LibRaw::strerror(rc));
    if (img->type != LIBRAW_IMAGE_BITMAP || img->bits != 16 || (img->colors != 3 && img->colors != 1)) {
      LibRaw::dcraw_clear_mem(img);
      return fail(out, "image", "unexpected pixel layout");
    }

    // with highlights kept, LibRaw scales by the largest white balance multiplier so nothing clips:
    // undo that here and the frame is as bright as a clipped one, with the highlights above 1.0
    const float *m = raw->imgdata.color.pre_mul;
    float lo = m[0] < m[1] ? m[0] : m[1];
    if (m[2] < lo) lo = m[2];
    out->gain = o->highlight > 0 && img->colors == 3 && lo > 0.01f && lo < 1.0f ? 1.0f / lo : 1.0f;
    out->handle = img;
    out->px = reinterpret_cast<const unsigned short *>(img->data);
    out->width = img->width;
    out->height = img->height;
    out->colors = img->colors;
    return 0;
  } catch (const std::exception &e) {
    if (img) LibRaw::dcraw_clear_mem(img);
    return fail(out, "LibRaw", e.what());
  } catch (...) {
    if (img) LibRaw::dcraw_clear_mem(img);
    return fail(out, "LibRaw", "unexpected failure");
  }
}

// LibRaw's matrix from the file's white-balanced camera RGB to linear sRGB (rgb_cam, from its metadata:
// no decoding). Fails for sensors that are not three-colour.
int ol_raw_camera_matrix(const unsigned char *data, size_t len, float out[9]) {
  try {
    std::unique_ptr<LibRaw> raw(new LibRaw);
    if (raw->open_buffer(data, len) != LIBRAW_SUCCESS || raw->imgdata.idata.colors != 3) return 1;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) out[i * 3 + j] = raw->imgdata.color.rgb_cam[i][j];
    return 0;
  } catch (...) {
    return 1;
  }
}

void ol_raw_free(void *handle) {
  if (handle) LibRaw::dcraw_clear_mem(static_cast<libraw_processed_image_t *>(handle));
}

int ol_parse_makernotes(
    const uint8_t *data,
    size_t len,
    OlMakerNotesOut *out
) {
  if (!out) return -1;
  std::memset(out, 0, sizeof(*out));
  if (!data || len < 16) return -1;

  // Scan binary TIFF/EXIF/MakerNote IFD for Nikon 0x00a7 shutter count (works on both RAW and JPEG)
  scan_file_shutter_count(data, len, &out->shutter_count);

  try {
    std::unique_ptr<OlRaw> raw(new OlRaw);
    int rc = raw->open_buffer(data, len);
    if (rc != LIBRAW_SUCCESS) {
      return out->shutter_count > 0 ? 0 : -1;
    }

    const auto &id = raw->imgdata.idata;
    const auto &lens = raw->imgdata.lens;
    const auto &mn = raw->imgdata.makernotes;
    const auto &col = raw->imgdata.color;
    const auto &sz = raw->imgdata.sizes;

    std::snprintf(out->camera_model, sizeof(out->camera_model), "%s %s", id.make, id.model);
    if (lens.LensSerial[0]) {
      std::snprintf(out->lens_serial, sizeof(out->lens_serial), "%s", lens.LensSerial);
    } else if (lens.InternalLensSerial[0]) {
      std::snprintf(out->lens_serial, sizeof(out->lens_serial), "%s", lens.InternalLensSerial);
    }
    if (raw->imgdata.shootinginfo.BodySerial[0]) {
      std::snprintf(out->body_serial, sizeof(out->body_serial), "%s", raw->imgdata.shootinginfo.BodySerial);
    } else if (raw->imgdata.shootinginfo.InternalBodySerial[0]) {
      std::snprintf(out->body_serial, sizeof(out->body_serial), "%s", raw->imgdata.shootinginfo.InternalBodySerial);
    }

    // Vendor ID
    if (ieq_prefix(id.make, "Canon")) out->vendor_id = 1;
    else if (ieq_prefix(id.make, "Nikon")) out->vendor_id = 2;
    else if (ieq_prefix(id.make, "Sony")) out->vendor_id = 3;
    else if (ieq_prefix(id.make, "Fuji")) out->vendor_id = 4;
    else if (ieq_prefix(id.make, "Olympus") || ieq_prefix(id.make, "OM Digital")) out->vendor_id = 5;
    else if (ieq_prefix(id.make, "Panasonic")) out->vendor_id = 6;
    else if (ieq_prefix(id.make, "Pentax") || ieq_prefix(id.make, "Ricoh")) out->vendor_id = 7;

    // Sensor CFA pattern
    out->is_xtrans = (id.filters == 9) ? 1 : 0;
    if (id.filters == 9) {
      std::snprintf(out->cfa_pattern, sizeof(out->cfa_pattern), "X-Trans 6x6");
    } else if (id.filters > 1000) {
      const char *names = "RGBG";
      char pat[5] = {
          names[raw->FC(0, 0) & 3],
          names[raw->FC(0, 1) & 3],
          names[raw->FC(1, 0) & 3],
          names[raw->FC(1, 1) & 3],
          '\0'
      };
      std::snprintf(out->cfa_pattern, sizeof(out->cfa_pattern), "%s", pat);
    } else if (id.colors == 1) {
      std::snprintf(out->cfa_pattern, sizeof(out->cfa_pattern), "Monochrome");
    } else {
      std::snprintf(out->cfa_pattern, sizeof(out->cfa_pattern), "Linear RGB");
    }

    // Unpack optical black border statistics & per-channel black/white levels when header did not set them
    if (col.maximum == 0 && (col.black + col.cblack[0]) == 0) {
      raw->unpack();
    }
    out->raw_width = sz.raw_width;
    out->raw_height = sz.raw_height;
    out->white_level = col.maximum;
    for (int c = 0; c < 4; ++c) {
      out->black_levels[c] = col.black + col.cblack[c];
      out->border_black_counts[c] = col.black_stat[4 + c];
      out->linear_max[c] = col.linear_max[c];
      out->cam_mul[c] = col.cam_mul[c];
    }

    // Vendor shutter counts
    if (out->shutter_count == 0) {
      if (mn.sony.ImageCount3 > 0 && mn.sony.ImageCount3 < 5000000) {
        out->shutter_count = mn.sony.ImageCount3;
      } else if (mn.fuji.ImageCount > 0 && mn.fuji.ImageCount < 5000000) {
        out->shutter_count = (uint32_t)mn.fuji.ImageCount;
      } else if (mn.sony.ShotNumberSincePowerUp > 0 && mn.sony.ShotNumberSincePowerUp < 5000000) {
        out->shutter_count = mn.sony.ShotNumberSincePowerUp;
      }
    }

    // Focus distance
    if (mn.olympus.FocusDistance > 0.0 && mn.olympus.FocusDistance < 10000.0) {
      out->focus_distance_m = (float)mn.olympus.FocusDistance;
    }

    // Decode AF points from vendor structs and `imCommon.afdata`
    // 1. Sony FocusLocation: [width, height, x, y]
    if (mn.sony.FocusLocation[0] > 0 && mn.sony.FocusLocation[1] > 0 &&
        mn.sony.FocusLocation[2] <= mn.sony.FocusLocation[0] &&
        mn.sony.FocusLocation[3] <= mn.sony.FocusLocation[1]) {
      float fx = (float)mn.sony.FocusLocation[2] / (float)mn.sony.FocusLocation[0];
      float fy = (float)mn.sony.FocusLocation[3] / (float)mn.sony.FocusLocation[1];
      uint8_t kind = mn.sony.AFTracking == 1 ? 2 : mn.sony.AFTracking == 2 ? 3 : 0; // face tracking, lock-on
      push_af(out, fx, fy, 0.045f, 0.045f, 1, kind);
    }

    // 2. Fujifilm FocusPixel: [x, y] in image coordinates
    if (out->af_count == 0 && mn.fuji.FocusPixel[0] > 0 && mn.fuji.FocusPixel[1] > 0 &&
        sz.width > 0 && sz.height > 0 &&
        mn.fuji.FocusPixel[0] <= sz.width && mn.fuji.FocusPixel[1] <= sz.height) {
      float fx = (float)mn.fuji.FocusPixel[0] / (float)sz.width;
      float fy = (float)mn.fuji.FocusPixel[1] / (float)sz.height;
      push_af(out, fx, fy, 0.045f, 0.045f, 1, 0);
    }

    // 3. Olympus AFPointSelected: normalized coordinates
    if (out->af_count == 0 && mn.olympus.AFPointSelected[0] > 0.0 && mn.olympus.AFPointSelected[0] < 1.0 &&
        mn.olympus.AFPointSelected[1] > 0.0 && mn.olympus.AFPointSelected[1] < 1.0) {
      push_af(out, (float)mn.olympus.AFPointSelected[0], (float)mn.olympus.AFPointSelected[1], 0.045f, 0.045f, 1, 0);
    }

    // 4. Canon & Nikon `imCommon.afdata[]`
    for (int i = 0; i < mn.common.afcount && out->af_count == 0; ++i) {
      const auto &af = mn.common.afdata[i];
      if (out->vendor_id == 1 && (af.AFInfoData_tag == 0x0026 || af.AFInfoData_tag == 0x003c)) {
        decode_canon_af(af, out);
      } else if (out->vendor_id == 2 && af.AFInfoData_tag == 0x00b7) {
        float f35 = raw->imgdata.lens.FocalLengthIn35mmFormat, f = raw->imgdata.other.focal_len;
        float crop = f35 > 0 && f > 0 ? f35 / f : raw->imgdata.lens.makernotes.CameraFormat == LIBRAW_FORMAT_APSC ? 1.5f : 1.f;
        decode_nikon_af(af, crop, out);
      }
    }
    orient_af(out, sz.flip);

    return 0;
  } catch (...) {
    return out->shutter_count > 0 ? 0 : -1;
  }
}

} // extern "C"
