// Native C/C++ lossless JPEG DCT transforms (libturbojpeg) and hardware-accelerated
// HEIC / HEIF / 10-bit AVIF container decoding (macOS ImageIO/CoreGraphics or libheif).

#include "studio_native.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef OL_HAVE_TURBOJPEG
#include <turbojpeg.h>
#endif

#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#endif

extern "C" {

void ol_free_buf(void *p) {
  if (p) std::free(p);
}

int ol_jpeg_lossless_transform(
    const uint8_t *jpeg_in,
    size_t len_in,
    int op,
    int crop_x,
    int crop_y,
    int crop_w,
    int crop_h,
    uint8_t **out_jpeg,
    size_t *out_len,
    char err_buf[160]
) {
  if (out_jpeg) *out_jpeg = nullptr;
  if (out_len) *out_len = 0;
  if (err_buf) err_buf[0] = '\0';
  if (!jpeg_in || len_in < 16 || !out_jpeg || !out_len) {
    if (err_buf) std::snprintf(err_buf, 160, "invalid JPEG input buffer");
    return -1;
  }

#ifdef OL_HAVE_TURBOJPEG
  tjhandle tj = tjInitTransform();
  if (!tj) {
    if (err_buf) std::snprintf(err_buf, 160, "tjInitTransform failed: %s", tjGetErrorStr());
    return -1;
  }

  int width = 0, height = 0, subsamp = 0, colorspace = 0;
  if (tjDecompressHeader3(tj, jpeg_in, (unsigned long)len_in, &width, &height, &subsamp, &colorspace) < 0) {
    if (err_buf) std::snprintf(err_buf, 160, "tjDecompressHeader3: %s", tjGetErrorStr2(tj));
    tjDestroy(tj);
    return -1;
  }

  tjtransform xform;
  std::memset(&xform, 0, sizeof(xform));
  xform.op = (op >= 0 && op <= 7) ? op : TJXOP_NONE;
  xform.options = TJXOPT_TRIM;

  if (crop_w > 0 && crop_h > 0) {
    int mcu_w = (subsamp >= 0 && subsamp < TJ_NUMSAMP) ? tjMCUWidth[subsamp] : 16;
    int mcu_h = (subsamp >= 0 && subsamp < TJ_NUMSAMP) ? tjMCUHeight[subsamp] : 16;
    if (mcu_w <= 0) mcu_w = 16;
    if (mcu_h <= 0) mcu_h = 16;
    // Align crop origin to MCU boundaries as required by lossless DCT cropping
    int ax = ( std::max(0, crop_x) / mcu_w ) * mcu_w;
    int ay = ( std::max(0, crop_y) / mcu_h ) * mcu_h;
    int aw = std::min(width - ax, std::max(mcu_w, crop_w));
    int ah = std::min(height - ay, std::max(mcu_h, crop_h));
    if (aw > 0 && ah > 0 && (ax > 0 || ay > 0 || aw < width || ah < height)) {
      xform.r.x = ax;
      xform.r.y = ay;
      xform.r.w = aw;
      xform.r.h = ah;
      xform.options |= TJXOPT_CROP;
    }
  }

  unsigned char *dst_buf = nullptr;
  unsigned long dst_size = 0;
  int rc = tjTransform(tj, jpeg_in, (unsigned long)len_in, 1, &dst_buf, &dst_size, &xform, 0);
  if (rc < 0 || !dst_buf || dst_size == 0) {
    if (err_buf) std::snprintf(err_buf, 160, "tjTransform: %s", tjGetErrorStr2(tj));
    if (dst_buf) tjFree(dst_buf);
    tjDestroy(tj);
    return -1;
  }

  uint8_t *copy = static_cast<uint8_t *>(std::malloc(dst_size));
  if (!copy) {
    tjFree(dst_buf);
    tjDestroy(tj);
    if (err_buf) std::snprintf(err_buf, 160, "out of memory");
    return -1;
  }
  std::memcpy(copy, dst_buf, dst_size);
  tjFree(dst_buf);
  tjDestroy(tj);
  *out_jpeg = copy;
  *out_len = (size_t)dst_size;
  return 0;
#else
  if (err_buf) std::snprintf(err_buf, 160, "libturbojpeg not linked in this build");
  return -2;
#endif
}

int ol_decode_heif_container(
    const uint8_t *data,
    size_t len,
    int max_edge,
    uint8_t **out_rgb,
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
  if (w == 0 || h == 0) {
    CGImageRelease(cg_img);
    CFRelease(src);
    CFRelease(cf_data);
    if (err_buf) std::snprintf(err_buf, 160, "empty HEIC/AVIF image dimensions");
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

  std::vector<uint8_t> rgba(w * h * 4, 0);
  CGContextRef ctx = CGBitmapContextCreate(
      rgba.data(),
      w,
      h,
      8,
      w * 4,
      render_cs,
      (uint32_t)kCGImageAlphaNoneSkipLast | (uint32_t)kCGBitmapByteOrder32Big
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

  uint8_t *rgb = static_cast<uint8_t *>(std::malloc(w * h * 3));
  if (!rgb) {
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
