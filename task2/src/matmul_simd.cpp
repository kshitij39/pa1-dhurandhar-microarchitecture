#include <immintrin.h>
#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {

    for (int i = 0; i < M; i++) {

        int j = 0;

        // Process 3 columns of B at a time
        for (; j + 2 < N; j += 3) {

            // One SIMD register for C.
            // Lane 0 -> C[i][j]
            // Lane 1 -> C[i][j+1]
            // Lane 2 -> C[i][j+2]
            // Lanes 3-7 are unused for now.
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

                // B is transposed, so its columns are contiguous.
                //
                // B column j     -> B[j][k ... k+7]
                // B column j+1   -> B[j+1][k ... k+7]
                // B column j+2   -> B[j+2][k ... k+7]
                __m256 b0 = _mm256_loadu_ps(
                    &B[j * ldb + k]
                );

                __m256 b1 = _mm256_loadu_ps(
                    &B[(j + 1) * ldb + k]
                );

                __m256 b2 = _mm256_loadu_ps(
                    &B[(j + 2) * ldb + k]
                );

                // Element-wise multiplication
                //
                // a × b0 -> partial result for C[i][j]
                // a × b1 -> partial result for C[i][j+1]
                // a × b2 -> partial result for C[i][j+2]
                __m256 product1 = _mm256_mul_ps(a, b0);
                __m256 product2 = _mm256_mul_ps(a, b1);
                __m256 product3 = _mm256_mul_ps(a, b2);

                // Store products temporarily so we can
                // calculate the dot product of each 8-element chunk.
                float temp1[8];
                float temp2[8];
                float temp3[8];

                _mm256_storeu_ps(temp1, product1);
                _mm256_storeu_ps(temp2, product2);
                _mm256_storeu_ps(temp3, product3);

                // Dot product for C[i][j]
                float dot1 =
                    temp1[0] + temp1[1] +
                    temp1[2] + temp1[3] +
                    temp1[4] + temp1[5] +
                    temp1[6] + temp1[7];

                // Dot product for C[i][j+1]
                float dot2 =
                    temp2[0] + temp2[1] +
                    temp2[2] + temp2[3] +
                    temp2[4] + temp2[5] +
                    temp2[6] + temp2[7];

                // Dot product for C[i][j+2]
                float dot3 =
                    temp3[0] + temp3[1] +
                    temp3[2] + temp3[3] +
                    temp3[4] + temp3[5] +
                    temp3[6] + temp3[7];

                // Put the three partial dot products into
                // three lanes of the C SIMD register.
                //
                // c[0] -> C[i][j]
                // c[1] -> C[i][j+1]
                // c[2] -> C[i][j+2]
                __m256 partial = _mm256_setzero_ps();

                partial[0] = dot1;
                partial[1] = dot2;
                partial[2] = dot3;

                // Accumulate this 8-element K chunk.
                c = _mm256_add_ps(c, partial);
            }

            // Handle remaining K elements (< 8)
            float remainder1 = 0.0f;
            float remainder2 = 0.0f;
            float remainder3 = 0.0f;

            for (; k < K; k++) {

                remainder1 +=
                    A[i * lda + k] *
                    B[j * ldb + k];

                remainder2 +=
                    A[i * lda + k] *
                    B[(j + 1) * ldb + k];

                remainder3 +=
                    A[i * lda + k] *
                    B[(j + 2) * ldb + k];
            }

            // Store the three C values
            C[i * ldc + j]     = c[0] + remainder1;
            C[i * ldc + j + 1] = c[1] + remainder2;
            C[i * ldc + j + 2] = c[2] + remainder3;
        }

        // ----------------------------------------------------
        // Cleanup: if N is not a multiple of 3
        // ----------------------------------------------------
        //
        // Example:
        // N = 10
        //
        // SIMD handles:
        // 0,1,2
        // 3,4,5
        // 6,7,8
        //
        // Cleanup handles:
        // 9
        //
        for (; j < N; j++) {

            float sum = 0.0f;

            for (int k = 0; k < K; k++) {

                sum +=
                    A[i * lda + k] *
                    B[j * ldb + k];
            }

            C[i * ldc + j] = sum;
        }
    }
}