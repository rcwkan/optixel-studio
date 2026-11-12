// HEIC / HEIF / 10-bit AVIF container decoding with macOS ImageIO, at 16 bits per channel.
// (Lossless JPEG transforms are the `turbojpeg` crate's: src/libraw.rs.)

#include "studio_native.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#endif

extern "C" {

void ol_free_buf(void *p) {
  if (p) std::free(p);
}

int ol_decode_heif_container(
    const uint8_t *data,
    size_t len,
    int max_edge,
    uint16_t **out_rgb,
    uint32_t *out_w,
    uint32_t *out_h,
    uint8_t **out_icc,
    size_t *out_icc_len,
    char err_buf[160]
) {
  if (out_rgb) *out_rgb = nullptr;
  if (out_w) *out_w = 0;
  if (out_h) *out_h = 0;
  if (out_icc) *out_icc = nullptr;
  if (out_icc_len) *out_icc_len = 0;
  if (err_buf) err_buf[0] = '\0';
  if (!data || len < 16 || !out_rgb || !out_w || !out_h) {
    if (err_buf) std::snprintf(err_buf, 160, "invalid HEIF/AVIF input buffer");
    return -1;
  }

#ifdef __APPLE__
  CFDataRef cf_data = CFDataCreateWithBytesNoCopy(kCFAllocatorDefault, data, (CFIndex)len, kCFAllocatorNull);
  if (!cf_data) {
    if (err_buf) std::snprintf(err_buf, 160, "CFDataCreateWithBytesNoCopy failed");
    return -1;
  }
  CGImageSourceRef src = CGImageSourceCreateWithData(cf_data, nullptr);
  if (!src) {
    CFRelease(cf_data);
    if (err_buf) std::snprintf(err_buf, 160, "CGImageSourceCreateWithData failed");
    return -1;
  }

  // Use CGImageSourceCreateThumbnailAtIndex with kCGImageSourceCreateThumbnailWithTransform = true
  // so HEIC container `irot`/`imir` and EXIF orientation are applied automatically.
  int target_max = max_edge > 0 ? max_edge : 16384;
  CFNumberRef cf_max = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &target_max);
  const void *keys[4] = {
      kCGImageSourceCreateThumbnailFromImageAlways,
      kCGImageSourceCreateThumbnailWithTransform,
      kCGImageSourceShouldCacheImmediately,
      kCGImageSourceThumbnailMaxPixelSize
  };
  const void *vals[4] = {
      kCFBooleanTrue,
      kCFBooleanTrue,
      kCFBooleanTrue,
      cf_max
  };
  CFDictionaryRef opts = CFDictionaryCreate(
      kCFAllocatorDefault, keys, vals, 4, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks
  );
  CGImageRef cg_img = CGImageSourceCreateThumbnailAtIndex(src, 0, opts);
  if (opts) CFRelease(opts);
  if (cf_max) CFRelease(cf_max);
  if (!cg_img) {
    cg_img = CGImageSourceCreateImageAtIndex(src, 0, nullptr);
  }
  if (!cg_img) {
    CFRelease(src);
    CFRelease(cf_data);
    if (err_buf) std::snprintf(err_buf, 160, "ImageIO could not decode HEIC/AVIF container");
    return -1;
  }

  size_t w = CGImageGetWidth(cg_img);
  size_t h = CGImageGetHeight(cg_img);
  if (w == 0 || h == 0 || w > 32768 || h > 32768 || (size_t)w > SIZE_MAX / h / 8) {
    CGImageRelease(cg_img);
    CFRelease(src);
    CFRelease(cf_data);
    if (err_buf) std::snprintf(err_buf, 160, "invalid HEIC/AVIF image dimensions");
    return -1;
  }

  // Extract embedded ICC profile (e.g., Apple Display P3) if present, and render in the image's own
  // RGB color space (or sRGB fallback) so wide-gamut P3 primaries are not clipped!
  CGColorSpaceRef img_cs = CGImageGetColorSpace(cg_img);
  CGColorSpaceRef render_cs = nullptr;
  if (img_cs && CGColorSpaceGetModel(img_cs) == kCGColorSpaceModelRGB) {
    render_cs = CGColorSpaceRetain(img_cs);
    if (out_icc && out_icc_len) {
      CFDataRef icc_data = CGColorSpaceCopyICCData(img_cs);
      if (icc_data) {
        size_t icc_len = (size_t)CFDataGetLength(icc_data);
        if (icc_len > 0) {
          uint8_t *icc_buf = static_cast<uint8_t *>(std::malloc(icc_len));
          if (icc_buf) {
            std::memcpy(icc_buf, CFDataGetBytePtr(icc_data), icc_len);
            *out_icc = icc_buf;
            *out_icc_len = icc_len;
          }
        }
        CFRelease(icc_data);
      }
    }
  } else {
    render_cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  }

  // 16 bits per channel: a 10-bit HEIC / AVIF keeps its 1024 levels (an 8-bit context bands them)
  std::vector<uint16_t> rgba(w * h * 4, 0);
  CGContextRef ctx = CGBitmapContextCreate(
      rgba.data(),
      w,
      h,
      16,
      w * 8,
      render_cs,
      (uint32_t)kCGImageAlphaNoneSkipLast | (uint32_t)kCGBitmapByteOrder16Little
  );
  if (render_cs) CGColorSpaceRelease(render_cs);
  if (!ctx) {
    CGImageRelease(cg_img);
    CFRelease(src);
    CFRelease(cf_data);
    if (err_buf) std::snprintf(err_buf, 160, "CGBitmapContextCreate failed");
    return -1;
  }

  CGContextDrawImage(ctx, CGRectMake(0, 0, (CGFloat)w, (CGFloat)h), cg_img);
  CGContextRelease(ctx);
  CGImageRelease(cg_img);
  CFRelease(src);
  CFRelease(cf_data);

  uint16_t *rgb = static_cast<uint16_t *>(std::malloc(w * h * 3 * sizeof(uint16_t)));
  if (!rgb) {
    if (out_icc && *out_icc) {
      std::free(*out_icc);
      *out_icc = nullptr;
      if (out_icc_len) *out_icc_len = 0;
    }
    if (err_buf) std::snprintf(err_buf, 160, "out of memory");
    return -1;
  }
  for (size_t i = 0; i < w * h; ++i) {
    rgb[i * 3 + 0] = rgba[i * 4 + 0];
    rgb[i * 3 + 1] = rgba[i * 4 + 1];
    rgb[i * 3 + 2] = rgba[i * 4 + 2];
  }
  *out_rgb = rgb;
  *out_w = (uint32_t)w;
  *out_h = (uint32_t)h;
  return 0;
#else
  (void)max_edge;
  if (err_buf) std::snprintf(err_buf, 160, "native HEIF decoder not available on this platform");
  return -2;
#endif
}

} // extern "C"
