// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING (CORRECTED)
#include <immintrin.h>
#include "matmul.h"

inline int mini(int a, int b) {
    return (a > b) ? b : a;
}

static float reduce_sum(__m256 sum){
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


const int BLOCK_M = 64;
const int BLOCK_N = 64;
const int BLOCK_K = 64;
const int PREFETCH_DISTANCE = 64; // increased since we now prefetch less often — see below

// B is K x N, stored column-major with leading dimension ldb:
// element (k, j) lives at B[j*ldb + k] (each column j is contiguous in k)

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {

    // i_blk / j_blk outer, k_blk inner: each C tile is fully accumulated
    // across all of K before being written once (fixes the original
    // "read-modify-write C once per k-block" bug).
    for(int i_blk = 0; i_blk < M; i_blk += BLOCK_M) {
        int i_end = mini(i_blk + BLOCK_M, M);

        for(int j_blk = 0; j_blk < N; j_blk += BLOCK_N) {
            int j_end = mini(j_blk + BLOCK_N, N);

            // ---- Main tiled region: 4 rows x 2 cols micro-kernel ----
            // Only 8 __m256 accumulators now (was 16 for 4x4), leaving
            // headroom in the 16-register YMM file for a0..a3 + b0,b1
            // without spilling to the stack (the spills showed up
            // clearly in `perf annotate` with the 4x4 version).
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

                    // scalar tail accumulator for this tile, across all k-blocks
                    float tail[4][2] = {{0}};

                    for(int k_blk = 0; k_blk < K; k_blk += BLOCK_K) {
                        int k_end = mini(k_blk + BLOCK_K, K);

                        int k = k_blk;
                        for(; k + 7 < k_end; k += 8) {

                            // Cut back to 2 prefetches/iteration (was 8).
                            // perf stat showed very low LLC miss rate, i.e.
                            // not memory-bandwidth-bound — the hardware
                            // prefetcher covers most of this once the
                            // access pattern is this regular; we prefetch
                            // just the first A row and first B col further
                            // ahead as a light nudge rather than saturating
                            // the load ports with prefetch traffic.
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

                        // K-tail (scalar) for this k-block, accumulated into
                        // 'tail' since C isn't written until the whole tile is done.
                        for(; k < k_end; k++) {
                            for(int ti = 0; ti < 4; ti++) {
                                for(int tj = 0; tj < 2; tj++) {
                                    tail[ti][tj] += A[(i+ti)*lda + k] * B[(j+tj)*ldb + k];
                                }
                            }
                        }
                    } // k_blk

                    // Write C once per element for this tile.
                    C[i*ldc + j]       = reduce_sum(sum00) + tail[0][0];
                    C[i*ldc + j+1]     = reduce_sum(sum01) + tail[0][1];

                    C[(i+1)*ldc + j]   = reduce_sum(sum10) + tail[1][0];
                    C[(i+1)*ldc + j+1] = reduce_sum(sum11) + tail[1][1];

                    C[(i+2)*ldc + j]   = reduce_sum(sum20) + tail[2][0];
                    C[(i+2)*ldc + j+1] = reduce_sum(sum21) + tail[2][1];

                    C[(i+3)*ldc + j]   = reduce_sum(sum30) + tail[3][0];
                    C[(i+3)*ldc + j+1] = reduce_sum(sum31) + tail[3][1];
                }

                // J-tail: leftover column in [j_blk, j_end) when the block
                // width isn't a multiple of 2. Sweeps full K directly.
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

            // I-tail: leftover rows in [i_blk, i_end) not covered by the
            // 4-wide i loop.
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
