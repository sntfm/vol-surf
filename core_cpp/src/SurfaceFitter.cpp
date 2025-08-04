#include "SurfaceFitter.h"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <limits>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace SurfaceFitter {
namespace SVI {

#if defined(__x86_64__)
void evaluate_batch_avx512(
    const float* k,
    const SurfaceFitter::SVI::SVIParams& params,
    float* out,
    int n
) {
    const __m512 a = _mm512_set1_ps(params.a);
    const __m512 b = _mm512_set1_ps(params.b);
    const __m512 rho = _mm512_set1_ps(params.rho);
    const __m512 m = _mm512_set1_ps(params.m);
    const __m512 sigma_sq = _mm512_set1_ps(params.sigma * params.sigma);

    for (int i = 0; i < n; i += 16) {
        __m512 k_vec = _mm512_loadu_ps(&k[i]);
        __m512 k_m = _mm512_sub_ps(k_vec, m);
        __m512 k_m_sq = _mm512_mul_ps(k_m, k_m);
        __m512 sqrt_term = _mm512_sqrt_ps(_mm512_add_ps(k_m_sq, sigma_sq));
        __m512 inner = _mm512_mul_ps(b, _mm512_add_ps(
            _mm512_mul_ps(rho, k_m),
            sqrt_term
        ));
        __m512 result = _mm512_add_ps(a, inner);
        _mm512_storeu_ps(&out[i], result);
    }
}
#endif


float SurfaceFitter::SVI::SVIFitter::evaluate(float k, const SurfaceFitter::SVI::SVIParams& params) {
    float k_m = k - params.m;
    float inner = params.b * (params.rho * k_m + 
        std::sqrt(k_m * k_m + params.sigma * params.sigma));
    return params.a + inner;
}

float SurfaceFitter::SVI::SVIFitter::evaluate_first_derivative(float k, const SurfaceFitter::SVI::SVIParams& params) {
    float k_m = k - params.m;
    float sqrt_term = std::sqrt(k_m * k_m + params.sigma * params.sigma);
    return params.b * (params.rho + k_m / sqrt_term);
}

float SurfaceFitter::SVI::SVIFitter::evaluate_second_derivative(float k, const SurfaceFitter::SVI::SVIParams& params) {
    float k_m = k - params.m;
    float sigma_sq = params.sigma * params.sigma;
    float sqrt_term = std::sqrt(k_m * k_m + sigma_sq);
    float cube_term = sqrt_term * sqrt_term * sqrt_term;
    return params.b * sigma_sq / cube_term;
}

bool SurfaceFitter::SVI::SVIParams::is_butterfly_arbitrage_free(float k) const {
    float k_m = k - m;
    float sigma_sq = sigma * sigma;
    float sqrt_term = std::sqrt(k_m * k_m + sigma_sq);
    float cube_term = sqrt_term * sqrt_term * sqrt_term;
    return (b * sigma_sq / cube_term) >= 0;
}

bool SurfaceFitter::SVI::SVIParams::is_calendar_arbitrage_free(float k, float T1, float T2) const {
    return true;
}

void SurfaceFitter::SVI::SVIFitter::enforce_arbitrage_free(
    SurfaceFitter::SVI::SVIParams& params,
    const flatbuffers::Vector<float>* moneyness
) {
    float min_second_deriv = std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < moneyness->size(); ++i) {
        float k = moneyness->Get(i);
        float second_deriv = evaluate_second_derivative(k, params);
        min_second_deriv = std::min(min_second_deriv, second_deriv);
    }
    
    if (min_second_deriv < 0) {
        params.sigma *= 1.1f;
        params.b *= 0.9f;
        enforce_bounds(params);
    }
}

void SurfaceFitter::SVI::SVIFitter::initial_guess(
    const flatbuffers::Vector<float>* moneyness,
    const flatbuffers::Vector<float>* ivs,
    SurfaceFitter::SVI::SVIParams& params
) {
    float mean_k = 0.0f, mean_iv = 0.0f;
    float var_k = 0.0f, var_iv = 0.0f;
    float cov_k_iv = 0.0f;
    int n = moneyness->size();

    for (int i = 0; i < n; ++i) {
        mean_k += moneyness->Get(i);
        mean_iv += ivs->Get(i);
    }
    mean_k /= n;
    mean_iv /= n;

    for (int i = 0; i < n; ++i) {
        float dk = moneyness->Get(i) - mean_k;
        float div = ivs->Get(i) - mean_iv;
        var_k += dk * dk;
        var_iv += div * div;
        cov_k_iv += dk * div;
    }
    var_k /= (n - 1);
    var_iv /= (n - 1);
    cov_k_iv /= (n - 1);

    params.a = mean_iv * mean_iv;
    params.m = mean_k;
    params.sigma = std::sqrt(var_k);
    params.rho = std::copysign(0.5f, cov_k_iv);
    params.b = std::sqrt(var_iv) / params.sigma;

    enforce_bounds(params);
    enforce_arbitrage_free(params, moneyness);
}

float SurfaceFitter::SVI::SVIFitter::objective(
    const SurfaceFitter::SVI::SVIParams& params,
    const flatbuffers::Vector<float>* moneyness,
    const flatbuffers::Vector<float>* ivs
) {
    float sum_sq_err = 0.0f;
    int n = moneyness->size();
    
#if defined(__x86_64__)
    alignas(64) float k_array[n];
    alignas(64) float var_array[n];
    
    for (int i = 0; i < n; ++i) {
        k_array[i] = moneyness->Get(i);
    }
    
    evaluate_batch_avx512(k_array, params, var_array, n);
    
    for (int i = 0; i < n; ++i) {
        float iv = ivs->Get(i);
        float err = var_array[i] - iv * iv;
        sum_sq_err += err * err;
    }
#else
    for (int i = 0; i < n; ++i) {
        float k = moneyness->Get(i);
        float iv = ivs->Get(i);
        float model_var = evaluate(k, params);
        float err = model_var - iv * iv;
        sum_sq_err += err * err;
    }
#endif
    
    return sum_sq_err / n;
}

void SurfaceFitter::SVI::SVIFitter::gradient(
    const SurfaceFitter::SVI::SVIParams& params,
    const flatbuffers::Vector<float>* moneyness,
    const flatbuffers::Vector<float>* ivs,
    float grad[5]
) {
    const float h = 1e-4f;
    SurfaceFitter::SVI::SVIParams p = params;
    float base = objective(params, moneyness, ivs);
    
    p = params; p.a += h;
    grad[0] = (objective(p, moneyness, ivs) - base) / h;
    
    p = params; p.b += h;
    grad[1] = (objective(p, moneyness, ivs) - base) / h;
    
    p = params; p.rho += h;
    grad[2] = (objective(p, moneyness, ivs) - base) / h;
    
    p = params; p.m += h;
    grad[3] = (objective(p, moneyness, ivs) - base) / h;
    
    p = params; p.sigma += h;
    grad[4] = (objective(p, moneyness, ivs) - base) / h;
}

void SurfaceFitter::SVI::SVIFitter::enforce_bounds(SurfaceFitter::SVI::SVIParams& params) {
    params.a = std::max(0.0001f, params.a);
    params.b = std::clamp(params.b, 0.0001f, 5.0f);
    params.rho = std::clamp(params.rho, -0.99f, 0.99f);
    params.sigma = std::max(0.0001f, params.sigma);
}

bool SurfaceFitter::SVI::SVIFitter::lm_step(
    SurfaceFitter::SVI::SVIParams& params,
    const flatbuffers::Vector<float>* moneyness,
    const flatbuffers::Vector<float>* ivs,
    float lambda
) {
    float grad[5];
    gradient(params, moneyness, ivs, grad);
    
    float current_obj = objective(params, moneyness, ivs);
    SurfaceFitter::SVI::SVIParams new_params = params;
    
    new_params.a -= lambda * grad[0];
    new_params.b -= lambda * grad[1];
    new_params.rho -= lambda * grad[2];
    new_params.m -= lambda * grad[3];
    new_params.sigma -= lambda * grad[4];
    
    enforce_bounds(new_params);
    enforce_arbitrage_free(new_params, moneyness);
    
    float new_obj = objective(new_params, moneyness, ivs);
    
    if (new_obj < current_obj) {
        params = new_params;
        return true;
    }
    return false;
}

bool SurfaceFitter::SVI::SVIFitter::fit_slice(
    const flatbuffers::Vector<float>* moneyness,
    const flatbuffers::Vector<float>* ivs,
    SurfaceFitter::SVI::SVIParams& params,
    float time_budget_micros
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    initial_guess(moneyness, ivs, params);
    
    float lambda = 0.1f;
    int max_iter = 50;
    float tol = 1e-6f;
    
    float prev_obj = objective(params, moneyness, ivs);
    
    for (int iter = 0; iter < max_iter; ++iter) {
        auto now = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
        if (elapsed >= time_budget_micros) {
            return false;
        }
        
        bool improved = lm_step(params, moneyness, ivs, lambda);
        float current_obj = objective(params, moneyness, ivs);
        
        if (improved) {
            lambda *= 1.5f;
            if (std::abs(current_obj - prev_obj) < tol) {
                return true;
            }
        } else {
            lambda *= 0.5f;
            if (lambda < 1e-10f) {
                return false;
            }
        }
        
        prev_obj = current_obj;
    }
    
    return true;
}

// Direct vector interface - optimized for performance
bool SurfaceFitter::SVI::SVIFitter::fit_slice_direct(
    const std::vector<float>& moneyness,
    const std::vector<float>& ivs,
    SurfaceFitter::SVI::SVIParams& params,
    float time_budget_micros
) {
    if (moneyness.size() != ivs.size() || moneyness.size() < 3) {
        return false;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Fast initial guess based on raw moments
    float mean_k = 0.0f, mean_iv = 0.0f;
    float var_k = 0.0f, var_iv = 0.0f;
    float cov_k_iv = 0.0f;
    int n = moneyness.size();

    for (int i = 0; i < n; ++i) {
        mean_k += moneyness[i];
        mean_iv += ivs[i];
    }
    mean_k /= n;
    mean_iv /= n;

    for (int i = 0; i < n; ++i) {
        float dk = moneyness[i] - mean_k;
        float div = ivs[i] - mean_iv;
        var_k += dk * dk;
        var_iv += div * div;
        cov_k_iv += dk * div;
    }
    var_k /= (n - 1);
    var_iv /= (n - 1);
    cov_k_iv /= (n - 1);

    params.a = mean_iv * mean_iv;
    params.m = mean_k;
    params.sigma = std::max(0.0001f, std::sqrt(var_k));
    params.rho = std::clamp(std::copysign(0.5f, cov_k_iv), -0.99f, 0.99f);
    params.b = std::clamp(std::sqrt(var_iv) / params.sigma, 0.0001f, 5.0f);
    
    // Fast objective function for vectors
    auto fast_objective = [&](const SurfaceFitter::SVI::SVIParams& p) -> float {
        float sum_sq_err = 0.0f;
        for (int i = 0; i < n; ++i) {
            float k_m = moneyness[i] - p.m;
            float model_var = p.a + p.b * (p.rho * k_m + 
                std::sqrt(k_m * k_m + p.sigma * p.sigma));
            float err = model_var - ivs[i] * ivs[i];
            sum_sq_err += err * err;
        }
        return sum_sq_err / n;
    };
    
    float lambda = 0.1f;
    int max_iter = 30;  // Reduced iterations for speed
    float tol = 1e-5f;   // Slightly relaxed tolerance
    
    float prev_obj = fast_objective(params);
    
    for (int iter = 0; iter < max_iter; ++iter) {
        auto now = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
        if (elapsed >= time_budget_micros) {
            return prev_obj < 1e-3f;  // Return success if error is reasonable
        }
        
        // Simple gradient descent step
        const float h = 1e-4f;
        SurfaceFitter::SVI::SVIParams new_params = params;
        
        // Compute gradient
        float base = fast_objective(params);
        
        SurfaceFitter::SVI::SVIParams p = params;
        p.a += h; float grad_a = (fast_objective(p) - base) / h;
        p = params; p.b += h; float grad_b = (fast_objective(p) - base) / h;
        p = params; p.rho += h; float grad_rho = (fast_objective(p) - base) / h;
        p = params; p.m += h; float grad_m = (fast_objective(p) - base) / h;
        p = params; p.sigma += h; float grad_sigma = (fast_objective(p) - base) / h;
        
        // Update parameters
        new_params.a -= lambda * grad_a;
        new_params.b -= lambda * grad_b;
        new_params.rho -= lambda * grad_rho;
        new_params.m -= lambda * grad_m;
        new_params.sigma -= lambda * grad_sigma;
        
        // Enforce bounds
        new_params.a = std::max(0.0001f, new_params.a);
        new_params.b = std::clamp(new_params.b, 0.0001f, 5.0f);
        new_params.rho = std::clamp(new_params.rho, -0.99f, 0.99f);
        new_params.sigma = std::max(0.0001f, new_params.sigma);
        
        float new_obj = fast_objective(new_params);
        
        if (new_obj < prev_obj) {
            params = new_params;
            lambda *= 1.2f;
            if (std::abs(new_obj - prev_obj) < tol) {
                return true;
            }
            prev_obj = new_obj;
        } else {
            lambda *= 0.5f;
            if (lambda < 1e-8f) {
                return prev_obj < 1e-3f;
            }
        }
    }
    
    return prev_obj < 1e-3f;
}
}
}