// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics (4x2 Micro-kernel)
#include <immintrin.h>
#include "matmul.h"

float reduce_sum(__m256 sum){
    __m128 low = _mm256_castps256_ps128(sum);
    __m128 high = _mm256_extractf128_ps(sum,1);
    low = _mm_add_ps(low,high);
    low = _mm_hadd_ps(low,low);
    low = _mm_hadd_ps(low,low);
    return _mm_cvtss_f32(low);
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {

    int i = 0;
    for (; i + 3 < M; i += 4) {

        int j = 0;
        // CHANGED: step by 2 columns of B
        for (; j + 1 < N; j += 2) {
            
            // initializes 8 avx256 accumulators with 0
            __m256 sum00 = _mm256_setzero_ps(), sum01 = _mm256_setzero_ps();
            __m256 sum10 = _mm256_setzero_ps(), sum11 = _mm256_setzero_ps();
            __m256 sum20 = _mm256_setzero_ps(), sum21 = _mm256_setzero_ps();
            __m256 sum30 = _mm256_setzero_ps(), sum31 = _mm256_setzero_ps();

            int k = 0;
            // vectorized part: only while a full 8-wide chunk of k remains
            for (; k + 7 < K; k += 8) {
                // loaded 8 floats from first four rows of A
                __m256 a0 = _mm256_loadu_ps(&A[i*lda + k]);
                __m256 a1 = _mm256_loadu_ps(&A[(i+1)*lda + k]);
                __m256 a2 = _mm256_loadu_ps(&A[(i+2)*lda + k]);
                __m256 a3 = _mm256_loadu_ps(&A[(i+3)*lda + k]);
                
                // loaded 8 floats from first two columns of B
                __m256 b0 = _mm256_loadu_ps(&B[j*ldb + k]);
                __m256 b1 = _mm256_loadu_ps(&B[(j+1)*ldb + k]);
                
                // FMA operations
                sum00 = _mm256_fmadd_ps(a0, b0, sum00); sum01 = _mm256_fmadd_ps(a0, b1, sum01);
                sum10 = _mm256_fmadd_ps(a1, b0, sum10); sum11 = _mm256_fmadd_ps(a1, b1, sum11);
                sum20 = _mm256_fmadd_ps(a2, b0, sum20); sum21 = _mm256_fmadd_ps(a2, b1, sum21);
                sum30 = _mm256_fmadd_ps(a3, b0, sum30); sum31 = _mm256_fmadd_ps(a3, b1, sum31);
            }

            // reduce the 8-wide accumulators to scalars
            float c00=reduce_sum(sum00), c01=reduce_sum(sum01);
            float c10=reduce_sum(sum10), c11=reduce_sum(sum11);
            float c20=reduce_sum(sum20), c21=reduce_sum(sum21);
            float c30=reduce_sum(sum30), c31=reduce_sum(sum31);

            // scalar tail: whatever's left of K that didn't fill a full 8-wide chunk
            for (; k < K; k++) {
                float a0=A[i*lda+k],     a1=A[(i+1)*lda+k];
                float a2=A[(i+2)*lda+k], a3=A[(i+3)*lda+k];
                float b0=B[j*ldb+k],     b1=B[(j+1)*ldb+k];

                c00+=a0*b0; c01+=a0*b1;
                c10+=a1*b0; c11+=a1*b1;
                c20+=a2*b0; c21+=a2*b1;
                c30+=a3*b0; c31+=a3*b1;
            }

            // Write results to C
            C[i*ldc+j]=c00;     C[i*ldc+j+1]=c01;
            C[(i+1)*ldc+j]=c10; C[(i+1)*ldc+j+1]=c11;
            C[(i+2)*ldc+j]=c20; C[(i+2)*ldc+j+1]=c21;
            C[(i+3)*ldc+j]=c30; C[(i+3)*ldc+j+1]=c31;
        }

        // remainder columns (N % 2 != 0): scalar, still 4 rows at a time
        // Note: Loop bounds condition left as is, it safely sweeps the remaining cols
        for (; j < N; j++) {
            float c0=0.f, c1=0.f, c2=0.f, c3=0.f;
            for (int k = 0; k < K; k++) {
                float b = B[j*ldb + k];
                c0 += A[i*lda+k]     * b;
                c1 += A[(i+1)*lda+k] * b;
                c2 += A[(i+2)*lda+k] * b;
                c3 += A[(i+3)*lda+k] * b;
            }
            C[i*ldc+j]=c0; C[(i+1)*ldc+j]=c1; C[(i+2)*ldc+j]=c2; C[(i+3)*ldc+j]=c3;
        }
    }

    // remainder rows (M % 4 != 0): fully scalar
    for (; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++)
                acc += A[i*lda + k] * B[j*ldb + k];
            C[i*ldc + j] = acc;
        }
    }
}
