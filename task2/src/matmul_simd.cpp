#include <immintrin.h>
#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {

    for (int i = 0; i < M; i++) {

        for (int j = 0; j < N; j++) {

            // SIMD register for C.
            // We only use lane 0 for C[i][j].
            // The other 7 lanes are intentionally unused for now.
            __m256 c = _mm256_setzero_ps();

            // Process K elements in chunks of 8
            int k = 0;

            for (; k + 7 < K; k += 8) {

                // Load 8 consecutive elements from row i of A
                //
                // A[i][k ... k+7]
                __m256 a = _mm256_loadu_ps(
                    &A[i * lda + k]
                );

                // B is assumed to be TRANSPOSED.
                //
                // Therefore B column j is stored contiguously.
                //
                // B[j][k ... k+7]
                __m256 b = _mm256_loadu_ps(
                    &B[j * ldb + k]
                );

                // Element-wise multiplication:
                //
                // [a0 a1 ... a7]
                //       ×
                // [b0 b1 ... b7]
                //
                // gives:
                // [a0b0 a1b1 ... a7b7]
                __m256 product = _mm256_mul_ps(a, b);

                // Horizontally add the 8 products
                // to obtain ONE dot-product value.
                //
                // result = a0b0 + a1b1 + ... + a7b7
                float temp[8];
                _mm256_storeu_ps(temp, product);

                float dot =
                    temp[0] + temp[1] +
                    temp[2] + temp[3] +
                    temp[4] + temp[5] +
                    temp[6] + temp[7];

                // Put this dot-product into lane 0.
                __m256 partial = _mm256_setzero_ps();
                partial[0] = dot;

                // Accumulate into C lane 0.
                c = _mm256_add_ps(c, partial);
            }

            // Handle remaining K elements (< 8)
            float remainder = 0.0f;

            for (; k < K; k++) {
                remainder +=
                    A[i * lda + k] *
                    B[j * ldb + k];   // B is transposed
            }

            // Extract lane 0 and add remainder.
            float c_value = c[0] + remainder;

            // Store final C[i][j]
            C[i * ldc + j] = c_value;
        }
    }
}