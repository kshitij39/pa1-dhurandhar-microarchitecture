#include <immintrin.h>
#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {

    for (int i = 0; i < M; i++) {

        int j = 0;

        // Process 4 columns of B at a time
        for (; j + 3 < N; j += 4) {

            // sum[0] -> C[i][j]
            // sum[1] -> C[i][j+1]
            // sum[2] -> C[i][j+2]
            // sum[3] -> C[i][j+3]
            __m256 sum = _mm256_setzero_ps();

            // Process K elements in chunks of 8
            int k = 0;

            for (; k + 7 < K; k += 8) {

                // Load 8 consecutive elements from A
                __m256 a = _mm256_loadu_ps(&A[i * lda + k]);

                // Load 8 elements from each of the 4 B columns
                __m256 b0 = _mm256_loadu_ps(&B[j * ldb + k]);

                __m256 b1 = _mm256_loadu_ps(&B[(j + 1) * ldb + k]);

                __m256 b2 = _mm256_loadu_ps(&B[(j + 2) * ldb + k]);

                __m256 b3 = _mm256_loadu_ps(&B[(j + 3) * ldb + k]);

                // Element-wise multiplication
                b0 = _mm256_mul_ps(a, b0);
                b1 = _mm256_mul_ps(a, b1);
                b2 = _mm256_mul_ps(a, b2);
                b3 = _mm256_mul_ps(a, b3);


                // ------------------------------------------------
                // Reduce b0: 8 SIMD values -> 1 dot product
                // ------------------------------------------------

                __m128 b0_low  = _mm256_castps256_ps128(b0);
                __m128 b0_high = _mm256_extractf128_ps(b0, 1);

                b0_low = _mm_add_ps(b0_low, b0_high);
                b0_low = _mm_hadd_ps(b0_low, b0_low);
                b0_low = _mm_hadd_ps(b0_low, b0_low);


                // ------------------------------------------------
                // Reduce b1: 8 SIMD values -> 1 dot product
                // ------------------------------------------------

                __m128 b1_low  = _mm256_castps256_ps128(b1);
                __m128 b1_high = _mm256_extractf128_ps(b1, 1);

                b1_low = _mm_add_ps(b1_low, b1_high);
                b1_low = _mm_hadd_ps(b1_low, b1_low);
                b1_low = _mm_hadd_ps(b1_low, b1_low);


                // ------------------------------------------------
                // Reduce b2: 8 SIMD values -> 1 dot product
                // ------------------------------------------------

                __m128 b2_low  = _mm256_castps256_ps128(b2);
                __m128 b2_high = _mm256_extractf128_ps(b2, 1);

                b2_low = _mm_add_ps(b2_low, b2_high);
                b2_low = _mm_hadd_ps(b2_low, b2_low);
                b2_low = _mm_hadd_ps(b2_low, b2_low);


                // ------------------------------------------------
                // Reduce b3: 8 SIMD values -> 1 dot product
                // ------------------------------------------------

                __m128 b3_low  = _mm256_castps256_ps128(b3);
                __m128 b3_high = _mm256_extractf128_ps(b3, 1);

                b3_low = _mm_add_ps(b3_low, b3_high);
                b3_low = _mm_hadd_ps(b3_low, b3_low);
                b3_low = _mm_hadd_ps(b3_low, b3_low);


                // ------------------------------------------------
                // Put the 4 dot products into one SIMD register
                //
                // sum[0] -> column j
                // sum[1] -> column j+1
                // sum[2] -> column j+2
                // sum[3] -> column j+3
                // ------------------------------------------------

                __m128 dots = _mm_set_ps(
                    _mm_cvtss_f32(b3_low),
                    _mm_cvtss_f32(b2_low),
                    _mm_cvtss_f32(b1_low),
                    _mm_cvtss_f32(b0_low)
                );

                __m256 partial = _mm256_castps128_ps256(dots);

                // Accumulate this K chunk
                sum = _mm256_add_ps(sum, partial);
            }


            // Handle remaining K elements (< 8)

            float remainder1 = 0.0f;
            float remainder2 = 0.0f;
            float remainder3 = 0.0f;
            float remainder4 = 0.0f;

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

                remainder4 +=
                    A[i * lda + k] *
                    B[(j + 3) * ldb + k];
            }


            // Store the four final C values

            C[i * ldc + j] =
                sum[0] + remainder1;

            C[i * ldc + j + 1] =
                sum[1] + remainder2;

            C[i * ldc + j + 2] =
                sum[2] + remainder3;

            C[i * ldc + j + 3] =
                sum[3] + remainder4;
        }


        // Cleanup for remaining columns

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