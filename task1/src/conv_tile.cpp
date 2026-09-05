// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"
#define TILEH 96
#define TILEW 96


void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int by = 0; by < H; by += TILEH) {
        for (int bx = 0; bx < W; bx += TILEW) {
            
            int ly = (by + TILEH < H) ? by + TILEH : H;
            int lx = (bx + TILEW < W) ? bx + TILEW : W;
            
            for (int oy = by; oy < ly; oy++) {
                for (int ox = bx; ox < lx; ox++) {
				    float acc = 0.0f;
				    for (int ky = 0; ky < K; ++ky) {
				        for (int kx = 0; kx < K; ++kx) {
				            acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
				        }
				    }
				    out[oy * W + ox] = acc;
				}
			}
        }
    }
}
