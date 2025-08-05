#include "SolverImpVol.h"

#include <omp.h>
#include <cmath>
#include <iostream>

#if defined(__x86_64__)
#include <immintrin.h>
// Add libmvec declarations
extern "C" {
    __m512 _ZGVeN16v_log(__m512);
    __m512 _ZGVeN16v_exp(__m512);
}
#endif

#if defined(__x86_64__)
constexpr int simd_width = 16; // AVX-512
#endif

constexpr float PI = 3.14159265358979323846f;

// Helper for normal CDF fast approximation
inline float norm_cdf_scalar(float x) {
    return 0.5f * std::erfc(-x * static_cast<float>(M_SQRT1_2));
}

namespace SolverImpVol {

float bs_price_scalar(char type, float S, float K, float T, float r, float sigma) {
    float sqrtT = std::sqrt(T);
    float d1 = (std::log(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrtT);
    float d2 = d1 - sigma * sqrtT;
    if (type == 'C')
        return S * norm_cdf_scalar(d1) - K * std::exp(-r * T) * norm_cdf_scalar(d2);
    else
        return K * std::exp(-r * T) * norm_cdf_scalar(-d2) - S * norm_cdf_scalar(-d1);
}

float bs_vega_scalar(float S, float K, float T, float r, float sigma) {
    float sqrtT = std::sqrt(T);
    float d1 = (std::log(S / K) + (r + 0.5f * sigma * sigma) * T) / (sigma * sqrtT);
    return S * sqrtT * (1.0f / std::sqrt(2.0f * M_PI)) * std::exp(-0.5f * d1 * d1);
}

// Jaeckel's approximation
// Initial guess function
float initial_guess_jaeckal(char type, float S, float K, float T, float r, float market_price) {
    float F = S * std::exp(r * T);
    float log_moneyness = std::log(F / K);
    float atm_guess = std::sqrt(2.0f * std::abs(log_moneyness) / T);
    if (std::abs(log_moneyness) < 0.05f) {
        return atm_guess;
    } else {
        float adjustment = 1.0f + 0.5f * log_moneyness;  // increase vol for ITM, decrease for OTM
        if (adjustment < 0.5f) adjustment = 0.5f;
        if (adjustment > 2.0f) adjustment = 2.0f;
        float guess = atm_guess * adjustment;
        if (guess < 0.01f) guess = 0.01f;
        return guess;
    }
}

// Householder iteration: single third-order step
float householder_improve(char type, float S, float K, float T, float r,
                          float price, float sigma0) {
    float pr = bs_price_scalar(type, S, K, T, r, sigma0);
    float vega = bs_vega_scalar(S, K, T, r, sigma0);
    float error = pr - price;
    float g = error / vega;
    float g2 = g * g, g3 = g2 * g;
    // coefficients for third-order correction term
    float d1 = (std::log(S/K) + (r + 0.5f*sigma0*sigma0)*T) / (sigma0*std::sqrt(T));
    float phi = std::exp(-0.5f * d1*d1) / std::sqrt(2.0f * PI);
    float v = vega;
    float v2 = v * v;
    // simplified Householder: sigma1 = sigma0 - g*(1 + 0.5*g*(v' / v))
    return sigma0 - g * (1.0f + 0.5f * g * ( (d1 * phi * std::sqrt(T)) / v ));
}

float implied_vol_jaeckel(char type, float S, float K, float T, float r, float market_price) {
    if (market_price <= 0 || T <= 0) return 0.0f;
    float sigma = initial_guess_jaeckal(type, S, K, T, r, market_price);
    // Two Householder correction steps
    sigma = householder_improve(type, S, K, T, r, market_price, sigma);
    sigma = householder_improve(type, S, K, T, r, market_price, sigma);
    return sigma > 0.0f ? sigma : 0.0f;
}

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
        out_iv[i] = implied_vol_jaeckel(type, S, K->Get(i), T, r->Get(i), price->Get(i));
    }
}

#if defined(__x86_64__)
// Abramowitz & Stegun 7.1.26, approximation for erfc(x) for positive x 
__m512 erfc_approx_avx512(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 p = _mm512_set1_ps(0.3275911f);

    // Coefficients
    const __m512 a1 = _mm512_set1_ps(0.254829592f);
    const __m512 a2 = _mm512_set1_ps(-0.284496736f);
    const __m512 a3 = _mm512_set1_ps(1.421413741f);
    const __m512 a4 = _mm512_set1_ps(-1.453152027f);
    const __m512 a5 = _mm512_set1_ps(1.061405429f);

    __m512 t = _mm512_div_ps(one, _mm512_add_ps(one, _mm512_mul_ps(p, x)));

    // Polynomial evaluation Horner's method
    __m512 poly = a5;
    poly = _mm512_fmadd_ps(poly, t, a4);
    poly = _mm512_fmadd_ps(poly, t, a3);
    poly = _mm512_fmadd_ps(poly, t, a2);
    poly = _mm512_fmadd_ps(poly, t, a1);

    __m512 exp_term = _mm512_exp_ps(_mm512_mul_ps(x, _mm512_sub_ps(_mm512_setzero_ps(), x)));

    __m512 erf_approx = _mm512_mul_ps(poly, t);
    erf_approx = _mm512_mul_ps(erf_approx, exp_term);

    return _mm512_sub_ps(one, erf_approx);
}

// Full erfc(x) for positive and negative x
__m512 erfc_full_avx512(__m512 x) {
    __mmask16 sign_mask = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OS);
    __m512 abs_x = _mm512_abs_ps(x);
    __m512 erfc_pos = erfc_approx_avx512(abs_x);
    return _mm512_mask_sub_ps(erfc_pos, sign_mask, _mm512_set1_ps(2.0f), erfc_pos);
}

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

        __mmask16 fallback_mask = 0; // Bitmask for fallback to scalar

        for (int iter = 0; iter < max_iter; ++iter) {
            __m512 sqrtT = _mm512_sqrt_ps(t);
            __m512 d1 = _mm512_div_ps(
                _mm512_add_ps(_ZGVeN16v_log(_mm512_div_ps(s, k)), _mm512_mul_ps(_mm512_add_ps(rr, _mm512_mul_ps(_mm512_set1_ps(0.5f), _mm512_mul_ps(sigma, sigma))), t)),
                _mm512_mul_ps(sigma, sqrtT)
            );
            __m512 d2 = _mm512_sub_ps(d1, _mm512_mul_ps(sigma, sqrtT));

            __m512 cdf_d1 = _mm512_mul_ps(_mm512_set1_ps(0.5f), erfc_full_avx512(_mm512_mul_ps(d1, _mm512_set1_ps(-M_SQRT1_2))));
            __m512 cdf_d2 = _mm512_mul_ps(_mm512_set1_ps(0.5f), erfc_full_avx512(_mm512_mul_ps(d2, _mm512_set1_ps(-M_SQRT1_2))));

            __m512 e_rt = _ZGVeN16v_exp(_mm512_mul_ps(_mm512_set1_ps(-1.0f), _mm512_mul_ps(rr, t)));

            __m512 call_val = _mm512_sub_ps(_mm512_mul_ps(s, cdf_d1), _mm512_mul_ps(k, _mm512_mul_ps(e_rt, cdf_d2)));
            __m512 put_val  = _mm512_sub_ps(_mm512_mul_ps(k, _mm512_mul_ps(e_rt, _mm512_sub_ps(_mm512_set1_ps(1.0f), cdf_d2))),
                                          _mm512_mul_ps(s, _mm512_sub_ps(_mm512_set1_ps(1.0f), cdf_d1)));

            __m512 bs_price_scalar = (type == 'C') ? call_val : put_val;
            __m512 diff = _mm512_sub_ps(bs_price_scalar, p);

            __m512 nd1 = _ZGVeN16v_exp(_mm512_mul_ps(_mm512_set1_ps(-0.5f), _mm512_mul_ps(d1, d1)));
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
                out_iv[i + j] = implied_vol_jaeckel(type, S, K->Get(i + j), T, r->Get(i + j), price->Get(i + j));
            } else {
                out_iv[i + j] = tmp[j];
            }
        }
    }
}
#endif

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

} // namespace SolverImpVol