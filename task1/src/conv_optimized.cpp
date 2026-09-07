// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"


void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
                    
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    int oy = 0;
    for (; oy + 2 < H; oy += 3) {
        int ox = 0;
        for (; ox + 31 < W; ox += 32) {

            __m256 sum00 = _mm256_setzero_ps();
            __m256 sum01 = _mm256_setzero_ps();
            __m256 sum02 = _mm256_setzero_ps();
            __m256 sum03 = _mm256_setzero_ps();

            __m256 sum10 = _mm256_setzero_ps();
            __m256 sum11 = _mm256_setzero_ps();
            __m256 sum12 = _mm256_setzero_ps();
            __m256 sum13 = _mm256_setzero_ps();

            __m256 sum20 = _mm256_setzero_ps();
            __m256 sum21 = _mm256_setzero_ps();
            __m256 sum22 = _mm256_setzero_ps();
            __m256 sum23 = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    __m256 weight = _mm256_set1_ps(ker[ky * K + kx]);

                    __m256 input00 = _mm256_loadu_ps(&in[(oy + ky) * in_stride + ox + kx]);
                    __m256 input01 = _mm256_loadu_ps(&in[(oy + ky) * in_stride + ox + kx + 8]);
                    __m256 input02 = _mm256_loadu_ps(&in[(oy + ky) * in_stride + ox + kx + 16]);
                    __m256 input03 = _mm256_loadu_ps(&in[(oy + ky) * in_stride + ox + kx + 24]);

                    __m256 input10 = _mm256_loadu_ps(&in[(oy + 1 + ky) * in_stride + ox + kx]);
                    __m256 input11 = _mm256_loadu_ps(&in[(oy + 1 + ky) * in_stride + ox + kx + 8]);
                    __m256 input12 = _mm256_loadu_ps(&in[(oy + 1 + ky) * in_stride + ox + kx + 16]);
                    __m256 input13 = _mm256_loadu_ps(&in[(oy + 1 + ky) * in_stride + ox + kx + 24]);

                    __m256 input20 = _mm256_loadu_ps(&in[(oy + 2 + ky) * in_stride + ox + kx]);
                    __m256 input21 = _mm256_loadu_ps(&in[(oy + 2 + ky) * in_stride + ox + kx + 8]);
                    __m256 input22 = _mm256_loadu_ps(&in[(oy + 2 + ky) * in_stride + ox + kx + 16]);
                    __m256 input23 = _mm256_loadu_ps(&in[(oy + 2 + ky) * in_stride + ox + kx + 24]);

                    sum00 = _mm256_fmadd_ps(input00, weight, sum00);
                    sum01 = _mm256_fmadd_ps(input01, weight, sum01);
                    sum02 = _mm256_fmadd_ps(input02, weight, sum02);
                    sum03 = _mm256_fmadd_ps(input03, weight, sum03);

                    sum10 = _mm256_fmadd_ps(input10, weight, sum10);
                    sum11 = _mm256_fmadd_ps(input11, weight, sum11);
                    sum12 = _mm256_fmadd_ps(input12, weight, sum12);
                    sum13 = _mm256_fmadd_ps(input13, weight, sum13);

                    sum20 = _mm256_fmadd_ps(input20, weight, sum20);
                    sum21 = _mm256_fmadd_ps(input21, weight, sum21);
                    sum22 = _mm256_fmadd_ps(input22, weight, sum22);
                    sum23 = _mm256_fmadd_ps(input23, weight, sum23);
                }
            }
            _mm256_storeu_ps(&out[oy * W + ox], sum00);
            _mm256_storeu_ps(&out[oy * W + ox + 8], sum01);
            _mm256_storeu_ps(&out[oy * W + ox + 16], sum02);
            _mm256_storeu_ps(&out[oy * W + ox + 24], sum03);

            _mm256_storeu_ps(&out[(oy + 1) * W + ox], sum10);
            _mm256_storeu_ps(&out[(oy + 1) * W + ox + 8], sum11);
            _mm256_storeu_ps(&out[(oy + 1) * W + ox + 16], sum12);
            _mm256_storeu_ps(&out[(oy + 1) * W + ox + 24], sum13);

            _mm256_storeu_ps(&out[(oy + 2) * W + ox], sum20);
            _mm256_storeu_ps(&out[(oy + 2) * W + ox + 8], sum21);
            _mm256_storeu_ps(&out[(oy + 2) * W + ox + 16], sum22);
            _mm256_storeu_ps(&out[(oy + 2) * W + ox + 24], sum23);
        }

        for (; ox < W; ++ox) {
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    float weight = ker[ky * K + kx];
                    sum0 += in[(oy + ky) * in_stride + ox + kx] * weight;
                    sum1 += in[(oy + 1 + ky) * in_stride + ox + kx] * weight;
                    sum2 += in[(oy + 2 + ky) * in_stride + ox + kx] * weight;
                }
            }
            out[oy * W + ox] = sum0;
            out[(oy + 1) * W + ox] = sum1;
            out[(oy + 2) * W + ox] = sum2;
        }
    }

    for (; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + ox + kx] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = acc;
        }
    }
}

