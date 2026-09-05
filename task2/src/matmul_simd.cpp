// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.

    for(int i=0;i<M;i++){
        
        // one row of a
        // processing 4 columns of b at a time
        // c will be like:
        // c[i][j]   = row i of a * column j of b
        // c[i][j+1] = row i of a * column j+1 of b
        // c[i][j+2] = row i of a * column j+2 of b
        // c[i][j+3] = row i of a * column j+3 of b
        
        for(int j=0;j<N;j+=4){

            // sum are temps for storing a block * b block and adding them to their previous values
            // i.e. sum0 = sum0 + one block of row i of a * one block of solumn j of b
            // sum are 8 values size and summated after k loop completes i.e after one complete column and row ends
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            __m256 sum2 = _mm256_setzero_ps();
            __m256 sum3 = _mm256_setzero_ps();
        
            for(int k=0;k<K;k+=8){

                // loading 8 elements from row i of a
                __m256 a = _mm256_loadu_ps(&A[i*lda + k]);

                // loading 8 elements from columns of b
                // b0 corresponds to column j 
                // b1 corresponds to column j+1
                // b2 corresponds to column j+2
                // b3 corresponds to column j+3
                // b stored in transpose so consecutive columns in memory
                __m256 b0 = _mm256_loadu_ps(&B[j*ldb + k]);
                __m256 b1 = _mm256_loadu_ps(&B[(j+1)*ldb + k]);
                __m256 b2 = _mm256_loadu_ps(&B[(j+2)*ldb + k]);
                __m256 b3 = _mm256_loadu_ps(&B[(j+3)*ldb + k]);


                //sum = sum + a*b
                //ex; sum0 = sum0 + a*b0
                
                sum0 = _mm256_fmadd_ps(a,b0,sum0);
                sum1 = _mm256_fmadd_ps(a,b1,sum1);
                sum2 = _mm256_fmadd_ps(a,b2,sum2);
                sum3 = _mm256_fmadd_ps(a,b3,sum3);

            }

            // k loop finished means all elements of row i of a 
            // and all elements of column j, j+1, j+2, j+3 of b are used
            // but our each sum has 8 different values now we need to 
            // sum all these 8 to get one number which is eventually one entry of c
            // sum0 accumulation gives entry c[i][j]
            // sum1 accumulation gives entry c[i][j+1] 
            // same sum2 and sum3

            // sum values are reduced as first 4 in low and last 4 in high. then vertical sum of low and high
            // ex: sum0 = [1,2,3,4,5,6,7,8] -> low[1,2,3,4] high->[5,6,7,8]
            // now vertical sum of high and low
            // [1+5, 2+6, 3+7, 4+8]
            // now horizontal addition
            // [(1+5) + (2+6), (3+7) + (4+8)]
            // now again horizontal sum which gives one last final value

            // first 4 of sum0
            __m128 sum0_low = _mm256_castps256_ps128(sum0);
            // last 4 of sum0
            __m128 sum0_high = _mm256_extractf128_ps(sum0,1);

            //verical sum
            sum0_low = _mm_add_ps(sum0_low,sum0_high);
            //horizontal sum
            sum0_low = _mm_hadd_ps(sum0_low, sum0_low);
            //again horizontal sum
            sum0_low = _mm_hadd_ps(sum0_low, sum0_low);
            //now one value in sum0_ low that is c[i][j]
            // same apply for sum1,2,3

            __m128 sum1_low = _mm256_castps256_ps128(sum1);
            __m128 sum1_high = _mm256_extractf128_ps(sum1,1);
            sum1_low = _mm_add_ps(sum1_low,sum1_high);
            sum1_low = _mm_hadd_ps(sum1_low, sum1_low);
            sum1_low = _mm_hadd_ps(sum1_low, sum1_low);

            __m128 sum2_low = _mm256_castps256_ps128(sum2);
            __m128 sum2_high = _mm256_extractf128_ps(sum2,1);
            sum2_low = _mm_add_ps(sum2_low,sum2_high);
            sum2_low = _mm_hadd_ps(sum2_low, sum2_low);
            sum2_low = _mm_hadd_ps(sum2_low, sum2_low);   
            
            __m128 sum3_low = _mm256_castps256_ps128(sum3);
            __m128 sum3_high = _mm256_extractf128_ps(sum3,1);
            sum3_low = _mm_add_ps(sum3_low,sum3_high);
            sum3_low = _mm_hadd_ps(sum3_low, sum3_low);
            sum3_low = _mm_hadd_ps(sum3_low, sum3_low);

            C[i*ldc + j]   = _mm_cvtss_f32(sum0_low);
            C[i*ldc + j+1] = _mm_cvtss_f32(sum1_low);
            C[i*ldc + j+2] = _mm_cvtss_f32(sum2_low);
            C[i*ldc + j+3] = _mm_cvtss_f32(sum3_low);

        }
    }

    //matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}