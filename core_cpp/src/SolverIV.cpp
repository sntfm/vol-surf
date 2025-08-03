#include "SolverIV.h"
#include <omp.h>
#include <cmath>
#include <iostream>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace {

#if defined(__x86_64__)
constexpr int simd_width = 16;
#endif
constexpr int max_iter = 10;
constexpr float epsilon = 1e-5f;

inline float norm_cdf_scalar(float x) {
    return 0.5f * std::erfc(-x * static_cast<float>(M_SQRT1_2));
}
} // anonymous namespace

namespace iv {

float black_scholes_price_scalar(char type, float S, float K, float T, float r, float sigma) {
    float sqrtT = std::sqrt(T);
    float d1 = (std::log(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrtT);
    float d2 = d1 - sigma * sqrtT;
    if (type == 'C')
        return S * norm_cdf_scalar(d1) - K * std::exp(-r * T) * norm_cdf_scalar(d2);
    else
        return K * std::exp(-r * T) * norm_cdf_scalar(-d2) - S * norm_cdf_scalar(-d1);
}

float vega_scalar(float S, float K, float T, float r, float sigma) {
    float sqrtT = std::sqrt(T);
    float d1 = (std::log(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrtT);
    return S * sqrtT * (1.0f / std::sqrt(2.0f * M_PI)) * std::exp(-0.5f * d1 * d1);
}

float implied_vol_brent(char type, float S, float K, float T, float r, float market_price) {
    float a = 1e-4f, b = 5.0f, fa, fb, fm;
    for (int i = 0; i < 50; ++i) {
        fa = black_scholes_price_scalar(type, S, K, T, r, a) - market_price;
        fb = black_scholes_price_scalar(type, S, K, T, r, b) - market_price;
        if (fa * fb >= 0) return 0.0f;
        float m = 0.5f * (a + b);
        fm = black_scholes_price_scalar(type, S, K, T, r, m) - market_price;
        if (std::abs(fm) < epsilon) return m;
        if (fa * fm < 0) b = m; else a = m;
    }
    return 0.0f;
}

#if defined(__x86_64__)
// AVX-512 implementation Newton-Raphson method for IV calculation
void compute_iv_avx512(
    const float& S,
    const flatbuffers::Vector<float>* K,
    const float& T,
    const flatbuffers::Vector<float>* r,
    const flatbuffers::Vector<float>* price,
    char type,
    std::vector<float>& out_iv
) {
    int N = K->size();
    out_iv.resize(N);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i += simd_width) {
        __m512 s     = _mm512_set1_ps(S);
        __m512 k     = _mm512_loadu_ps(&K->Get(i));
        __m512 t     = _mm512_set1_ps(T);
        __m512 rr    = _mm512_loadu_ps(&r->Get(i));
        __m512 p     = _mm512_loadu_ps(&price->Get(i));
        __m512 sigma = _mm512_set1_ps(0.2f);

        __mmask16 fallback_mask = 0;

        for (int iter = 0; iter < max_iter; ++iter) {
            __m512 sqrtT = _mm512_sqrt_ps(t);
            __m512 d1 = _mm512_div_ps(
                _mm512_add_ps(_mm512_log_ps(_mm512_div_ps(s, k)), _mm512_mul_ps(_mm512_add_ps(rr, _mm512_mul_ps(_mm512_set1_ps(0.5f), _mm512_mul_ps(sigma, sigma))), t)),
                _mm512_mul_ps(sigma, sqrtT)
            );
            __m512 d2 = _mm512_sub_ps(d1, _mm512_mul_ps(sigma, sqrtT));

            __m512 cdf_d1 = _mm512_mul_ps(_mm512_set1_ps(0.5f), _mm512_erfc_ps(_mm512_mul_ps(d1, _mm512_set1_ps(-M_SQRT1_2f))));
            __m512 cdf_d2 = _mm512_mul_ps(_mm512_set1_ps(0.5f), _mm512_erfc_ps(_mm512_mul_ps(d2, _mm512_set1_ps(-M_SQRT1_2f))));

            __m512 e_rt = _mm512_exp_ps(_mm512_mul_ps(_mm512_set1_ps(-1.0f), _mm512_mul_ps(rr, t)));

            __m512 call_val = _mm512_sub_ps(_mm512_mul_ps(s, cdf_d1), _mm512_mul_ps(k, _mm512_mul_ps(e_rt, cdf_d2)));
            __m512 put_val  = _mm512_sub_ps(_mm512_mul_ps(k, _mm512_mul_ps(e_rt, _mm512_sub_ps(_mm512_set1_ps(1.0f), cdf_d2))),
                                          _mm512_mul_ps(s, _mm512_sub_ps(_mm512_set1_ps(1.0f), cdf_d1)));

            __m512 bs_price = (type == 'C') ? call_val : put_val;
            __m512 diff = _mm512_sub_ps(bs_price, p);

            __m512 nd1 = _mm512_exp_ps(_mm512_mul_ps(_mm512_set1_ps(-0.5f), _mm512_mul_ps(d1, d1)));
            __m512 vega = _mm512_mul_ps(s, _mm512_mul_ps(sqrtT, _mm512_div_ps(nd1, _mm512_set1_ps(std::sqrt(2.0f * M_PI)))));

            __mmask16 small_vega = _mm512_cmp_ps_mask(vega, _mm512_set1_ps(1e-8f), _CMP_LT_OQ);
            __mmask16 bad_sigma  = _mm512_kor(
                _mm512_cmp_ps_mask(sigma, _mm512_set1_ps(1e-4f), _CMP_LT_OQ),
                _mm512_cmp_ps_mask(sigma, _mm512_set1_ps(5.0f), _CMP_GT_OQ)
            );

            fallback_mask |= (small_vega | bad_sigma);

            __m512 sigma_adj = _mm512_div_ps(diff, vega);
            sigma = _mm512_sub_ps(sigma, sigma_adj);

            __mmask16 converged = _mm512_cmp_ps_mask(_mm512_abs_ps(diff), _mm512_set1_ps(epsilon), _CMP_LT_OQ);
            if (converged == 0xFFFF) break;
        }

        float tmp[simd_width];
        _mm512_storeu_ps(tmp, sigma);

        for (int j = 0; j < simd_width && i + j < N; ++j) {
            if (fallback_mask & (1 << j)) {
                out_iv[i + j] = implied_vol_brent(type, S, K->Get(i + j), T, r->Get(i + j), price->Get(i + j));
            } else {
                out_iv[i + j] = tmp[j];
            }
        }
    }
}
#endif

// Fallback scalar implementation
void compute_iv_scalar(
    const float& S,
    const flatbuffers::Vector<float>* K,
    const float& T,
    const flatbuffers::Vector<float>* r,
    const flatbuffers::Vector<float>* price,
    char type,
    std::vector<float>& out_iv
) {
    int N = K->size();
    out_iv.resize(N);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        out_iv[i] = implied_vol_brent(type, S, K->Get(i), T, r->Get(i), price->Get(i));
    }
}

// Main entry point that chooses the appropriate implementation
void compute_iv(
    const float& S,
    const flatbuffers::Vector<float>* K,
    const float& T,
    const flatbuffers::Vector<float>* r,
    const flatbuffers::Vector<float>* price,
    char type,
    std::vector<float>& out_iv
) {
#if defined(__x86_64__)
    #ifdef HAS_AVX512
    compute_iv_avx512(S, K, T, r, price, type, out_iv);
    #else
    compute_iv_scalar(S, K, T, r, price, type, out_iv);
    #endif
#else
    compute_iv_scalar(S, K, T, r, price, type, out_iv);
#endif
}

} // namespace iv


