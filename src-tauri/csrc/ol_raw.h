// Optixel Studio's LibRaw subclass: the same decoder, with three clean-room demosaics of our own
// (RCD, LMMSE, AMaZE) plugged into LibRaw's Bayer interpolation step. LibRaw still does everything
// around it: optical black subtraction, white balance, scaling before, highlight recovery and the
// camera matrix after.
#pragma once

#include "libraw/libraw.h"
#include "studio_native.h"

// Our demosaic codes, above LibRaw's own user_qual numbers (0-12).
enum {
  OL_DEMOSAIC_RCD = 101,
  OL_DEMOSAIC_LMMSE = 102,
  OL_DEMOSAIC_AMAZE = 103
};

class OlRaw : public LibRaw {
public:
  // 0: LibRaw's own demosaic; else one of the codes above (Bayer sensors only: X-Trans, Foveon and
  // 4-colour sensors keep LibRaw's).
  int ol_demosaic = 0;

  void use_own_demosaic(int code) {
    ol_demosaic = code;
    callbacks.interpolate_bayer_cb = code ? &OlRaw::interpolate : nullptr;
  }

  // Extract the 2x2 Bayer pattern (0 R, 1 G, 2 B) in image coordinates.
  // Returns false when the sensor is not a standard 3-color Bayer mosaic.
  bool bayer_2x2(int pat[2][2]) {
    if (imgdata.idata.filters <= 1000 || imgdata.idata.colors > 4) return false;
    int seen[3] = {0, 0, 0};
    for (int r = 0; r < 2; ++r) {
      for (int c = 0; c < 2; ++c) {
        int col = FC(r, c);
        if (col == 3) col = 1; // G2 -> G
        if (col < 0 || col > 2) return false;
        pat[r][c] = col;
        seen[col]++;
      }
    }
    return seen[0] == 1 && seen[1] == 2 && seen[2] == 1;
  }

private:
  static void interpolate(void *self);
};
