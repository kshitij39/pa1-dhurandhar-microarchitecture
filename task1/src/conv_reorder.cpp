#include "convolution.h"

void conv_reorder(const float* in, float* out, const float* ker,int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    
    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            out[oy * W + ox] = 0.0f;
        }
    }

    
    for (int ky = 0; ky < K; ++ky) {

        const int input_row_offset = ky * in_stride;
        const int kernel_row_offset = ky * K;

        for (int kx = 0; kx < K; ++kx) {

            const float weight = ker[kernel_row_offset + kx];
            const int input_col_offset = kx;

            for (int oy = 0; oy < H; ++oy) {

                const int out_row = oy * W;
                const int in_row = (oy * in_stride)+ input_row_offset + input_col_offset;

                for (int ox = 0; ox < W; ++ox) {

                    out[out_row + ox] +=
                        in[in_row + ox] * weight;
                }
            }
        }
    }
}