// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING (CORRECTED)
#include <immintrin.h>
#include "matmul.h"

inline int mini(int a, int b) {
    return (a > b) ? b : a;
}

static float reduce_sum(__m256 sum){
    
    __m128 low = _mm256_castps256_ps128(sum);
    
    __m128 high = _mm256_extractf128_ps(sum,1);

    
    low = _mm_add_ps(low,high);
    
    low = _mm_hadd_ps(low,low);
    low = _mm_hadd_ps(low,low);

    
    return _mm_cvtss_f32(low);
}


const int BLOCK_M = 128;
const int BLOCK_N = 128;
const int BLOCK_K = 128;
const int PREFETCH_DISTANCE = 64; 



void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {

    
    for(int i_blk = 0; i_blk < M; i_blk += BLOCK_M) {
        int i_end = mini(i_blk + BLOCK_M, M);

        for(int j_blk = 0; j_blk < N; j_blk += BLOCK_N) {
            int j_end = mini(j_blk + BLOCK_N, N);

           
            for(int i = i_blk; i + 3 < i_end; i += 4) {
                for(int j = j_blk; j + 1 < j_end; j += 2) {

                    __m256 sum00 = _mm256_setzero_ps();
                    __m256 sum01 = _mm256_setzero_ps();

                    __m256 sum10 = _mm256_setzero_ps();
                    __m256 sum11 = _mm256_setzero_ps();

                    __m256 sum20 = _mm256_setzero_ps();
                    __m256 sum21 = _mm256_setzero_ps();

                    __m256 sum30 = _mm256_setzero_ps();
                    __m256 sum31 = _mm256_setzero_ps();


                    float tail[4][2] = {{0}};

                    for(int k_blk = 0; k_blk < K; k_blk += BLOCK_K) {
                        int k_end = mini(k_blk + BLOCK_K, K);

                        int k = k_blk;
                        for(; k + 7 < k_end; k += 8) {

                           
                            _mm_prefetch((const char*)&A[i*lda + k + PREFETCH_DISTANCE], _MM_HINT_T0);
                            _mm_prefetch((const char*)&B[j*ldb + k + PREFETCH_DISTANCE], _MM_HINT_T0);

                            __m256 a0 = _mm256_loadu_ps(&A[i*lda + k]);
                            __m256 a1 = _mm256_loadu_ps(&A[(i+1)*lda + k]);
                            __m256 a2 = _mm256_loadu_ps(&A[(i+2)*lda + k]);
                            __m256 a3 = _mm256_loadu_ps(&A[(i+3)*lda + k]);

                            __m256 b0 = _mm256_loadu_ps(&B[j*ldb + k]);
                            __m256 b1 = _mm256_loadu_ps(&B[(j+1)*ldb + k]);

                            sum00 = _mm256_fmadd_ps(a0, b0, sum00);
                            sum01 = _mm256_fmadd_ps(a0, b1, sum01);

                            sum10 = _mm256_fmadd_ps(a1, b0, sum10);
                            sum11 = _mm256_fmadd_ps(a1, b1, sum11);

                            sum20 = _mm256_fmadd_ps(a2, b0, sum20);
                            sum21 = _mm256_fmadd_ps(a2, b1, sum21);

                            sum30 = _mm256_fmadd_ps(a3, b0, sum30);
                            sum31 = _mm256_fmadd_ps(a3, b1, sum31);
                        }

                        
                        for(; k < k_end; k++) {
                            for(int ti = 0; ti < 4; ti++) {
                                for(int tj = 0; tj < 2; tj++) {
                                    tail[ti][tj] += A[(i+ti)*lda + k] * B[(j+tj)*ldb + k];
                                }
                            }
                        }
                    } 


                    C[i*ldc + j]       = reduce_sum(sum00) + tail[0][0];
                    C[i*ldc + j+1]     = reduce_sum(sum01) + tail[0][1];

                    C[(i+1)*ldc + j]   = reduce_sum(sum10) + tail[1][0];
                    C[(i+1)*ldc + j+1] = reduce_sum(sum11) + tail[1][1];

                    C[(i+2)*ldc + j]   = reduce_sum(sum20) + tail[2][0];
                    C[(i+2)*ldc + j+1] = reduce_sum(sum21) + tail[2][1];

                    C[(i+3)*ldc + j]   = reduce_sum(sum30) + tail[3][0];
                    C[(i+3)*ldc + j+1] = reduce_sum(sum31) + tail[3][1];
                }

               
                for(int j = j_blk + ((j_end - j_blk) / 2) * 2; j < j_end; j++) {
                    for(int ti = 0; ti < 4; ti++) {
                        float acc = 0.0f;
                        for(int k = 0; k < K; k++) {
                            acc += A[(i+ti)*lda + k] * B[j*ldb + k];
                        }
                        C[(i+ti)*ldc + j] = acc;
                    }
                }
            }

           
            for(int i = i_blk + ((i_end - i_blk) / 4) * 4; i < i_end; i++) {
                for(int j = j_blk; j < j_end; j++) {
                    float acc = 0.0f;
                    for(int k = 0; k < K; k++) {
                        acc += A[i*lda + k] * B[j*ldb + k];
                    }
                    C[i*ldc + j] = acc;
                }
            }
        }
    }
}
