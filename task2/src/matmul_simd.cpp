#include <immintrin.h>
#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {

    for (int i = 0; i < M; i++) {

        int j = 0;

        // Process 5 columns of B at a time
        for (; j + 3 < N; j += 4) {

            // One SIMD register for C.
            // Lane 0 -> C[i][j]
            // Lane 1 -> C[i][j+1]
            // Lane 2 -> C[i][j+2]
            // Lane 3 -> C[i][j+3]
            // Lane 4 -> C[i][j+4]
            // Lane 5 -> C[i][j+5]
            // Lane 6 -> C[i][j+6]
            // Lane 7 -> C[i][j+7]
            
            __m256 c = _mm256_setzero_ps();

            // Process K elements in chunks of 8
            int k = 0;

            for (; k + 7 < K; k += 8) {

                // Load 8 consecutive elements from row i of A
                //
                // A[i][k ... k+7]
                __m256 a = _mm256_loadu_ps(&A[i * lda + k]);

                // B is transposed, so its columns are contiguous.
                //
                // B column j     -> B[j][k ... k+7]
                // B column j+1   -> B[j+1][k ... k+7]
                // B column j+2   -> B[j+2][k ... k+7]
                // B column j+3   -> B[j+3][k ... k+7]
                // B column j+4   -> B[j+4][k ... k+7]
                // B column j+5   -> B[j+5][k ... k+7]
                // B column j+6   -> B[j+6][k ... k+7]
                // B column j+7   -> B[j+7][k ... k+7]
                __m256 b0 = _mm256_loadu_ps(&B[j * ldb + k]);
                __m256 b1 = _mm256_loadu_ps(&B[(j + 1) * ldb + k]);
                __m256 b2 = _mm256_loadu_ps(&B[(j + 2) * ldb + k]);
                __m256 b3 = _mm256_loadu_ps(&B[(j + 3) * ldb + k]);
                //__m256 b4 = _mm256_loadu_ps(&B[(j + 4) * ldb + k]);
                // __m256 b5 = _mm256_loadu_ps(&B[(j + 5) * ldb + k]);
                // __m256 b6 = _mm256_loadu_ps(&B[(j + 6) * ldb + k]);
                // __m256 b7 = _mm256_loadu_ps(&B[(j + 7) * ldb + k]);
                // Element-wise multiplication
                //
                // a × b0 -> partial result for C[i][j]
                // a × b1 -> partial result for C[i][j+1]
                // a × b2 -> partial result for C[i][j+2]
                // a × b3 -> partial result for C[i][j+3]
                // a × b4 -> partial result for C[i][j+4]
                // a × b5 -> partial result for C[i][j+5]
                // a × b6 -> partial result for C[i][j+6]
                // a × b7 -> partial result for C[i][j+7]
                b0 = _mm256_mul_ps(a, b0);
                b1 = _mm256_mul_ps(a, b1);
                b2 = _mm256_mul_ps(a, b2);
                b3 = _mm256_mul_ps(a, b3);
                //b4 = _mm256_mul_ps(a, b4);
                //b5 = _mm256_mul_ps(a, b5);
                // b6 = _mm256_mul_ps(a, b6);
                // b7 = _mm256_mul_ps(a, b7);
                // Store products temporarily so we can
                // calculate the dot product of each 8-element chunk.
                float temp1[8];
                float temp2[8];
                float temp3[8];
                float temp4[8];
                //float temp5[8];
                //float temp6[8];
                // float temp7[8];
                // float temp8[8];

                _mm256_storeu_ps(temp1, b0);
                _mm256_storeu_ps(temp2, b1);
                _mm256_storeu_ps(temp3, b2);
                _mm256_storeu_ps(temp4, b3);
                //_mm256_storeu_ps(temp5, b4);
                //_mm256_storeu_ps(temp6, b5);
                // _mm256_storeu_ps(temp7, b6);
                // _mm256_storeu_ps(temp8, b7);

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

                // // Dot product for C[i][j+3]
                float dot4 =
                    temp4[0] + temp4[1] +
                    temp4[2] + temp4[3] +
                    temp4[4] + temp4[5] +
                    temp4[6] + temp4[7];

                // // Dot product for C[i][j+4]
                // float dot5 =
                //     temp5[0] + temp5[1] +
                //     temp5[2] + temp5[3] +
                //     temp5[4] + temp5[5] +
                //     temp5[6] + temp5[7];

                // Dot product for C[i][j+5]
                // float dot6 =
                //     temp6[0] + temp6[1] +
                //     temp6[2] + temp6[3] +
                //     temp6[4] + temp6[5] +
                //     temp6[6] + temp6[7];

                // Dot product for C[i][j+6]
                // float dot7 =
                //     temp7[0] + temp7[1] +
                //     temp7[2] + temp7[3] +
                //     temp7[4] + temp7[5] +
                //     temp7[6] + temp7[7];

                // // Dot product for C[i][j+7]
                // float dot8 =
                //     temp8[0] + temp8[1] +
                //     temp8[2] + temp8[3] +
                //     temp8[4] + temp8[5] +
                //     temp8[6] + temp8[7];

                // Put the four partial dot products into
                // four lanes of the C SIMD register.
                //
                // c[0] -> C[i][j]
                // c[1] -> C[i][j+1]
                // c[2] -> C[i][j+2]
                // c[3] -> C[i][j+3]
                __m256 partial = _mm256_setzero_ps();

                partial[0] = dot1;
                partial[1] = dot2;
                partial[2] = dot3;
                partial[3] = dot4;
                //partial[4] = dot5;
                //partial[5] = dot6;
                // partial[6] = dot7;
                // partial[7] = dot8;

                // Accumulate this 8-element K chunk.
                c = _mm256_add_ps(c, partial);
            }

            // Handle remaining K elements (< 8)
            float remainder1 = 0.0f;
            float remainder2 = 0.0f;
            float remainder3 = 0.0f;
            float remainder4 = 0.0f;
            //float remainder5 = 0.0f;
            //float remainder6 = 0.0f;
            // float remainder7 = 0.0f;
            // float remainder8 = 0.0f;

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

                // remainder5 +=
                //     A[i * lda + k] *
                //     B[(j + 4) * ldb + k];

                // remainder6 +=
                //     A[i * lda + k] *
                //     B[(j + 5) * ldb + k];

                // remainder7 +=
                //     A[i * lda + k] *
                //     B[(j + 6) * ldb + k];

                // remainder8 +=
                //     A[i * lda + k] *
                //     B[(j + 7) * ldb + k];
            }

            // Store the four C values
            C[i * ldc + j]     = c[0] + remainder1;
            C[i * ldc + j + 1] = c[1] + remainder2;
            C[i * ldc + j + 2] = c[2] + remainder3;
            C[i * ldc + j + 3] = c[3] + remainder4;
            //C[i * ldc + j + 4] = c[4] + remainder5;
            //C[i * ldc + j + 5] = c[5] + remainder6;
            // C[i * ldc + j + 6] = c[6] + remainder7;
            // C[i * ldc + j + 7] = c[7] + remainder8;
        }

        // ----------------------------------------------------
        // Cleanup: if N is not a multiple of 4
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