// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"
inline float horizontal_sum(__m256 v) {
	__m128 lo = _mm256_castps256_ps128(v);
	__m128 hi = _mm256_extractf128_ps(v, 1);
	__m128 sum128 = _mm_add_ps(lo, hi);
	__m128 shuf = _mm_movehdup_ps(sum128);
	__m128 sum2 = _mm_add_ps(sum128, shuf);
	__m128 shuf2 = _mm_movehl_ps(shuf, sum2);
	__m128 final_sum = _mm_add_ss(sum2, shuf2);
	return _mm_cvtss_f32(final_sum);
}
int mini(int a,int b)
{
	return (a>b)?b:a;
}
const int BLOCK_M=64;
const int BLOCK_N=64;
const int PREFETCH_DISTANCE =32;
void matmul_prefetch(const float* A, const float* B, float* C,
		int M, int N, int K, int lda, int ldb, int ldc) {
	// TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
	// implementation.
	for(int i_blk=0;i_blk<M ; i_blk+=BLOCK_M)
	{
		int i_end=mini(i_blk+BLOCK_M,M);

		for(int j_blk=0;j_blk<N;j_blk+=BLOCK_N)
		{
			int j_end=mini(j_blk+BLOCK_N,N);
			for(int i=i_blk;i<i_end;i=i+1){
				const float* a=A +static_cast<long>(i)*lda;
				int j=j_blk;
				for(;j+8<=j_end;j=j+8){
					//allocate 8 accumulators
					__m256 acc0=_mm256_setzero_ps();
					__m256 acc1=_mm256_setzero_ps();
					__m256 acc2=_mm256_setzero_ps();
					__m256 acc3=_mm256_setzero_ps();
					__m256 acc4=_mm256_setzero_ps();
					__m256 acc5=_mm256_setzero_ps();
					__m256 acc6=_mm256_setzero_ps();
					__m256 acc7=_mm256_setzero_ps();
					//initialise to store 8 float values of a
					__m256 ar=_mm256_setzero_ps();
					__m256 ar2=_mm256_setzero_ps();
					//to store 8 float values of b
					const float *b0=B + static_cast<long>(j+0)*ldb;
					const float *b1=B + static_cast<long>(j+1)*ldb;
					const float *b2=B + static_cast<long>(j+2)*ldb;
					const float *b3=B + static_cast<long>(j+3)*ldb;
					const float *b4=B + static_cast<long>(j+4)*ldb;
					const float *b5=B + static_cast<long>(j+5)*ldb;
					const float *b6=B + static_cast<long>(j+6)*ldb;
					const float *b7=B + static_cast<long>(j+7)*ldb;


					int p=0;
					for(;p<=K-16;p=p+16){

						//prefetching
						//only prefetch the 8float of a
						_mm_prefetch((const char*)(a+p+PREFETCH_DISTANCE),_MM_HINT_T0);
						/*
						_mm_prefetch((const char*)(b0+p+PREFETCH_DISTANCE),_MM_HINT_T0);
						_mm_prefetch((const char*)(b1+p+PREFETCH_DISTANCE),_MM_HINT_T0);
						_mm_prefetch((const char*)(b2+p+PREFETCH_DISTANCE),_MM_HINT_T0);
						_mm_prefetch((const char*)(b3+p+PREFETCH_DISTANCE),_MM_HINT_T0);
						_mm_prefetch((const char*)(b4+p+PREFETCH_DISTANCE),_MM_HINT_T0);
						_mm_prefetch((const char*)(b5+p+PREFETCH_DISTANCE),_MM_HINT_T0);
						_mm_prefetch((const char*)(b6+p+PREFETCH_DISTANCE),_MM_HINT_T0);
						_mm_prefetch((const char*)(b7+p+PREFETCH_DISTANCE),_MM_HINT_T0);
						*/

						ar=_mm256_loadu_ps(a+p);
						acc0=_mm256_fmadd_ps(ar,_mm256_loadu_ps(b0+p),acc0);
						acc1=_mm256_fmadd_ps(ar,_mm256_loadu_ps(b1+p),acc1);
						acc2=_mm256_fmadd_ps(ar,_mm256_loadu_ps(b2+p),acc2);
						acc3=_mm256_fmadd_ps(ar,_mm256_loadu_ps(b3+p),acc3);
						acc4=_mm256_fmadd_ps(ar,_mm256_loadu_ps(b4+p),acc4);
						acc5=_mm256_fmadd_ps(ar,_mm256_loadu_ps(b5+p),acc5);
						acc6=_mm256_fmadd_ps(ar,_mm256_loadu_ps(b6+p),acc6);
						acc7=_mm256_fmadd_ps(ar,_mm256_loadu_ps(b7+p),acc7);
						
						ar2=_mm256_loadu_ps(a+p+8);
						acc0=_mm256_fmadd_ps(ar2,_mm256_loadu_ps(b0+p+8),acc0);
						acc1=_mm256_fmadd_ps(ar2,_mm256_loadu_ps(b1+p+8),acc1);
						acc2=_mm256_fmadd_ps(ar2,_mm256_loadu_ps(b2+p+8),acc2);
						acc3=_mm256_fmadd_ps(ar2,_mm256_loadu_ps(b3+p+8),acc3);
						acc4=_mm256_fmadd_ps(ar2,_mm256_loadu_ps(b4+p+8),acc4);
						acc5=_mm256_fmadd_ps(ar2,_mm256_loadu_ps(b5+p+8),acc5);
						acc6=_mm256_fmadd_ps(ar2,_mm256_loadu_ps(b6+p+8),acc6);
						acc7=_mm256_fmadd_ps(ar2,_mm256_loadu_ps(b7+p+8),acc7);
					}
					float final_c_value0=horizontal_sum(acc0);
					float final_c_value1=horizontal_sum(acc1);
					float final_c_value2=horizontal_sum(acc2);
					float final_c_value3=horizontal_sum(acc3);
					float final_c_value4=horizontal_sum(acc4);
					float final_c_value5=horizontal_sum(acc5);
					float final_c_value6=horizontal_sum(acc6);
					float final_c_value7=horizontal_sum(acc7);
					for(;p<K;p++){
						final_c_value0+=a[p]*b0[p];
						final_c_value1+=a[p]*b1[p];
						final_c_value2+=a[p]*b2[p];
						final_c_value3+=a[p]*b3[p];
						final_c_value4+=a[p]*b4[p];
						final_c_value5+=a[p]*b5[p];
						final_c_value6+=a[p]*b6[p];
						final_c_value7+=a[p]*b7[p];
					}
					C[static_cast<long>(i) *ldc +(j+0)]=final_c_value0;
					C[static_cast<long>(i) *ldc +(j+1)]=final_c_value1;
					C[static_cast<long>(i) *ldc +(j+2)]=final_c_value2;
					C[static_cast<long>(i) *ldc +(j+3)]=final_c_value3;
					C[static_cast<long>(i) *ldc +(j+4)]=final_c_value4;
					C[static_cast<long>(i) *ldc +(j+5)]=final_c_value5;
					C[static_cast<long>(i) *ldc +(j+6)]=final_c_value6;
					C[static_cast<long>(i) *ldc +(j+7)]=final_c_value7;

				}
				for(;j<j_end;j++)
				{
					float final_c_value=0.0f;
					const float* b_tail=B +static_cast<long>(j) *ldb;
					for(int p=0;p<K;p++)
					{
						final_c_value+=a[p]*b_tail[p];
					}
					C[static_cast<long>(i)*ldc +j]=final_c_value;
				}


			}


		}

	}


	//matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}
