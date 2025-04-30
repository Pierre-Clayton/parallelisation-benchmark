#include "attention_impl.h"
#include <algorithm>
#include <immintrin.h>
#include <omp.h>

// Accès p(i,j) dans matrice plate
inline float atf(const float* M, int cols, int i, int j) {
    return M[i*cols + j];
}
inline double atd(const double* M, int cols, int i, int j) {
    return M[i*cols + j];
}

// Version de base (no AVX)
template<typename T>
void attention_base(int N, int D, int DV,
                    const T* Q, const T* K, const T* V,
                    T* out) {
    const T scale = T(1) / std::sqrt((T)D);

    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<N; ++i) {
        // 1. QK^T
        std::vector<T> scores(N);
        T maxv = -std::numeric_limits<T>::infinity();
        for(int j=0; j<N; ++j) {
            T sum = 0;
            for(int d=0; d<D; ++d) {
                sum += Q[i*D + d] * K[j*D + d];
            }
            scores[j] = sum * scale;
            if(scores[j] > maxv) maxv = scores[j];
        }
        // 2. softmax (stable)
        T ssum = 0;
        for(int j=0; j<N; ++j) {
            scores[j] = std::exp(scores[j] - maxv);
            ssum += scores[j];
        }
        // 3. pondération V
        for(int d=0; d<DV; ++d) out[i*DV + d] = 0;
        for(int j=0; j<N; ++j) {
            T w = scores[j] / ssum;
            for(int d=0; d<DV; ++d) {
                out[i*DV + d] += w * V[j*DV + d];
            }
        }
    }
}

// AVX2 accéléré (float)
void attention_avx_float(int N, int D, int DV,
                         const float* Q, const float* K, const float* V,
                         float* out) {
    const float scale = 1.0f / std::sqrt((float)D);
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<N; ++i) {
        // scores
        std::vector<float> scores(N);
        float maxv = -std::numeric_limits<float>::infinity();
        // Q·K
        for(int j=0; j<N; ++j) {
            __m256 sum_vec = _mm256_setzero_ps();
            int d=0;
            for(; d+7<D; d+=8) {
                __m256 qv = _mm256_loadu_ps(&Q[i*D + d]);
                __m256 kv = _mm256_loadu_ps(&K[j*D + d]);
                sum_vec = _mm256_fmadd_ps(qv, kv, sum_vec);
            }
            float sum = 0;
            float tmp[8]; _mm256_storeu_ps(tmp, sum_vec);
            for(int t=0; t<8; ++t) sum += tmp[t];
            for(; d<D; ++d) sum += Q[i*D + d] * K[j*D + d];
            scores[j] = sum * scale;
            if(scores[j] > maxv) maxv = scores[j];
        }
        // softmax
        float ssum = 0;
        for(int j=0; j<N; ++j) {
            float ex = std::exp(scores[j] - maxv);
            scores[j] = ex;
            ssum += ex;
        }
        // pondéré V
        for(int d=0; d<DV; ++d) out[i*DV + d] = 0;
        for(int j=0; j<N; ++j) {
            float w = scores[j] / ssum;
            __m256 wv = _mm256_set1_ps(w);
            int d=0;
            for(; d+7<DV; d+=8) {
                __m256 vv = _mm256_loadu_ps(&V[j*DV + d]);
                __m256 ov = _mm256_loadu_ps(&out[i*DV + d]);
                ov = _mm256_fmadd_ps(wv, vv, ov);
                _mm256_storeu_ps(&out[i*DV + d], ov);
            }
            for(; d<DV; ++d) {
                out[i*DV + d] += w * V[j*DV + d];
            }
        }
    }
}

// Même approche pour double (non vectorisée pour la démo)
// Dispatch

void attention_impl_cpp(int N, int D, int DV,
                       const float* Q, const float* K, const float* V,
                       float* out,
                       int block_size,
                       int version) {
    if(version == 0) attention_base<float>(N, D, DV, Q, K, V, out);
    else attention_avx_float(N, D, DV, Q, K, V, out);
}

void attention_impl_cpp(int N, int D, int DV,
                       const double* Q, const double* K, const double* V,
                       double* out,
                       int block_size,
                       int version) {
    attention_base<double>(N, D, DV, Q, K, V, out);
}