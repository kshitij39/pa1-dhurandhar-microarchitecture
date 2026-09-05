// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"


float reduce_sum(__m256 sum){
    // sum values are reduced as first 4 in low and last 4 in high. then vertical sum of low and high
    // ex: sum0 = [1,2,3,4,5,6,7,8] -> low[1,2,3,4] high->[5,6,7,8]
    // now vertical sum of high and low
    // [1+5, 2+6, 3+7, 4+8]
    // now horizontal addition
    // [(1+5) + (2+6), (3+7) + (4+8)]
    // now again horizontal sum which gives one last final value

    // get first 4 values
    __m128 low = _mm256_castps256_ps128(sum);

    // get last 4 values
    __m128 high = _mm256_extractf128_ps(sum,1);

    // vertical sum
    low = _mm_add_ps(low,high);
    // horizontal sum
    low = _mm_hadd_ps(low,low);
    // again horizontal sum
    low = _mm_hadd_ps(low,low);
    
    // extracting final value from low
    return _mm_cvtss_f32(low);

}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.

    for(int i=0;i+3<M;i+=4){
        
        // four row of a
        // 4 columns of b 
        // c will be like a block of 4*4
        // c[i][j]   c[i][j+1]   c[i][j+2]   c[i][j+3]
        // c[i+1][j] c[i+1][j+1] c[i+1][j+2] c[i+1][j+3]
        // c[i+2][j] c[i+2][j+1] c[i+2][j+2] c[i+2][j+3]
        // c[i+3][j] c[i+3][j+1] c[i+3][j+2] c[i+3][j+3]

        // c[i][j]   = row i of a * column j of b
        
        
        for(int j=0;j<N;j+=4){

            // sum are temps for storing a block * b block and adding them to their previous values
            // i.e. sum0 = sum0 + one block of row i of a * one block of solumn j of b
            // sum are 8 values size and summated after k loop completes i.e after one complete column and row ends
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

            __m256 sum30 = _mm256_setzero_ps();
            __m256 sum31 = _mm256_setzero_ps();
            __m256 sum32 = _mm256_setzero_ps();
            __m256 sum33 = _mm256_setzero_ps();

        
            for(int k=0;k<K;k+=8){

                // loading 8 elements from row i of a
                __m256 a0 = _mm256_loadu_ps(&A[i*lda + k]);
                __m256 a1 = _mm256_loadu_ps(&A[(i+1)*lda + k]);
                __m256 a2 = _mm256_loadu_ps(&A[(i+2)*lda + k]);
                __m256 a3 = _mm256_loadu_ps(&A[(i+3)*lda + k]);

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


                //sum =sum + a*b
                //ex; sum0 = sum0 + a*b0
                
                sum00 = _mm256_fmadd_ps(a0,b0,sum00);
                sum01 = _mm256_fmadd_ps(a0,b1,sum01);
                sum02 = _mm256_fmadd_ps(a0,b2,sum02);
                sum03 = _mm256_fmadd_ps(a0,b3,sum03);

                sum10 = _mm256_fmadd_ps(a1,b0,sum10);
                sum11 = _mm256_fmadd_ps(a1,b1,sum11);
                sum12 = _mm256_fmadd_ps(a1,b2,sum12);
                sum13 = _mm256_fmadd_ps(a1,b3,sum13);

                sum20 = _mm256_fmadd_ps(a2,b0,sum20);
                sum21 = _mm256_fmadd_ps(a2,b1,sum21);
                sum22 = _mm256_fmadd_ps(a2,b2,sum22);
                sum23 = _mm256_fmadd_ps(a2,b3,sum23);

                sum30 = _mm256_fmadd_ps(a3,b0,sum30);
                sum31 = _mm256_fmadd_ps(a3,b1,sum31);
                sum32 = _mm256_fmadd_ps(a3,b2,sum32);
                sum33 = _mm256_fmadd_ps(a3,b3,sum33);

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
            // for this using function reduce sum
            

            C[i*ldc + j]   = reduce_sum(sum00);
            C[i*ldc + j+1] = reduce_sum(sum01);
            C[i*ldc + j+2] = reduce_sum(sum02);
            C[i*ldc + j+3] = reduce_sum(sum03);

            C[(i+1)*ldc + j]   = reduce_sum(sum10);
            C[(i+1)*ldc + j+1] = reduce_sum(sum11);
            C[(i+1)*ldc + j+2] = reduce_sum(sum12);
            C[(i+1)*ldc + j+3] = reduce_sum(sum13);

            C[(i+2)*ldc + j]   = reduce_sum(sum20);
            C[(i+2)*ldc + j+1] = reduce_sum(sum21);
            C[(i+2)*ldc + j+2] = reduce_sum(sum22);
            C[(i+2)*ldc + j+3] = reduce_sum(sum23);

            C[(i+3)*ldc + j]   = reduce_sum(sum30);
            C[(i+3)*ldc + j+1] = reduce_sum(sum31);
            C[(i+3)*ldc + j+2] = reduce_sum(sum32);
            C[(i+3)*ldc + j+3] = reduce_sum(sum33);


        }
    }

    //matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}
