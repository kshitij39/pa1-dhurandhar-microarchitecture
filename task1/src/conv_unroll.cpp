#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ox += 8) {

            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const int in_row = (oy + ky) * in_stride + ox;
                const int ker_row = ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    const float weight = ker[ker_row + kx];

                    acc0 += in[in_row + kx]     * weight;
                    acc1 += in[in_row + kx + 1] * weight;
                    acc2 += in[in_row + kx + 2] * weight;
                    acc3 += in[in_row + kx + 3] * weight;
                    acc4 += in[in_row + kx + 4] * weight;
                    acc5 += in[in_row + kx + 5] * weight;
                    acc6 += in[in_row + kx + 6] * weight;
                    acc7 += in[in_row + kx + 7] * weight;
                }
            }

            out[oy * W + ox]     = acc0;
            out[oy * W + ox + 1] = acc1;
            out[oy * W + ox + 2] = acc2;
            out[oy * W + ox + 3] = acc3;
            out[oy * W + ox + 4] = acc4;
            out[oy * W + ox + 5] = acc5;
            out[oy * W + ox + 6] = acc6;
            out[oy * W + ox + 7] = acc7;
        }
    }
}