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

void Fitter::enforce_bounds(Params& params) {
    params.a = std::max(0.0001f, params.a);
    params.b = std::clamp(params.b, 0.0001f, 2.0f);
    params.rho = std::clamp(params.rho, -0.999f, 0.999f);
    params.sigma = std::clamp(params.sigma, 0.001f, 2.0f);
}

void Fitter::initial_guess(
    const std::vector<float>& moneyness,
    const std::vector<float>& ivs,
    Params& params
) {
    if (moneyness.empty() || ivs.empty()) {
        return;
    }
    
    int n = moneyness.size();
    float mean_k = 0.0f, mean_iv = 0.0f;
    for (int i = 0; i < n; ++i) {
        mean_k += moneyness[i];
        mean_iv += ivs[i];
    }
    mean_k /= n;
    mean_iv /= n;
    
    float var_k = 0.0f, var_iv = 0.0f;
    for (int i = 0; i < n; ++i) {
        float dk = moneyness[i] - mean_k;
        float div = ivs[i] - mean_iv;
        var_k += dk * dk;
        var_iv += div * div;
    }
    var_k /= (n - 1);
    var_iv /= (n - 1);
    
    // Set initial parameters based on data
    params.a = std::max(0.01f, mean_iv * mean_iv * 0.8f); 
    params.m = std::clamp(mean_k, -2.0f, 2.0f);
    params.sigma = std::clamp(std::sqrt(var_k) + 0.1f, 0.05f, 1.0f);
    params.rho = 0.0f;
    params.b = std::clamp(std::sqrt(var_iv) * 0.5f, 0.1f, 1.0f);
    
    enforce_bounds(params);
}

float Fitter::evaluate_variance(float k, const Params& params) {
    float k_m = k - params.m;
    float sqrt_term = std::sqrt(k_m * k_m + params.sigma * params.sigma);
    return params.a + params.b * (params.rho * k_m + sqrt_term);
}

float Fitter::evaluate_iv(float k, const Params& params) {
    float variance = evaluate_variance(k, params);
    return std::sqrt(std::max(0.0001f, variance));
}

float Fitter::objective_scalar(
    const Params& params,
    const std::vector<float>& moneyness,
    const std::vector<float>& ivs
) {
    float sum_sq_error = 0.0f;
    int n = moneyness.size();
    
    for (int i = 0; i < n; ++i) {
        float model_iv = evaluate_iv(moneyness[i], params);
        float error = model_iv - ivs[i];
        sum_sq_error += error * error;
    }
    
    return sum_sq_error / n;
}

bool Fitter::optimize_scalar(
    const std::vector<float>& moneyness,
    const std::vector<float>& ivs,
    Params& params,
    float time_budget_micros
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    float learning_rate = 0.01f;
    int max_iterations = 150;
    float prev_obj = objective_scalar(params, moneyness, ivs);
    float tolerance = std::max(1e-5f, prev_obj * 0.01f);  // Adaptive tolerance based on initial error
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        auto now = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
        if (elapsed >= time_budget_micros) {
            break;
        }
        
        // Compute gradients using finite differences
        const float h = 1e-4f;
        Params temp_params = params;
        
        temp_params.a += h;
        float grad_a = (objective_scalar(temp_params, moneyness, ivs) - prev_obj) / h;
        
        temp_params = params;
        temp_params.b += h;
        float grad_b = (objective_scalar(temp_params, moneyness, ivs) - prev_obj) / h;
        
        temp_params = params;
        temp_params.rho += h;
        float grad_rho = (objective_scalar(temp_params, moneyness, ivs) - prev_obj) / h;
        
        temp_params = params;
        temp_params.m += h;
        float grad_m = (objective_scalar(temp_params, moneyness, ivs) - prev_obj) / h;
        
        temp_params = params;
        temp_params.sigma += h;
        float grad_sigma = (objective_scalar(temp_params, moneyness, ivs) - prev_obj) / h;
        
        // Update parameters
        Params new_params = params;
        new_params.a -= learning_rate * grad_a;
        new_params.b -= learning_rate * grad_b;
        new_params.rho -= learning_rate * grad_rho;
        new_params.m -= learning_rate * grad_m;
        new_params.sigma -= learning_rate * grad_sigma;
        
        enforce_bounds(new_params);
        
        float new_obj = objective_scalar(new_params, moneyness, ivs);
        
        if (new_obj < prev_obj) {
            params = new_params;
            learning_rate *= 1.1f;
            
            if (std::abs(new_obj - prev_obj) < tolerance) {
                return true;
            }
            prev_obj = new_obj;
        } else {
            learning_rate *= 0.7f;  
            if (learning_rate < 1e-6f) {
                break;
            }
        }
    }
    
    return prev_obj < 0.01f;
}

float Fitter::evaluate_variance_second_derivative(float k, const Params& params) {
    float k_m = k - params.m;
    float denominator = std::pow(k_m * k_m + params.sigma * params.sigma, 1.5f);
    return params.b * params.sigma * params.sigma / denominator;
}

bool Fitter::is_butterfly_arb_free(const Params& params, const std::vector<float>& moneyness) {
    // Quick check: for SVI butterfly condition requires b ≥ 0
    if (params.b < 0) {
        return false;
    }
    
    // Detailed check: d2w/dk2 ≥ 0 for all strikes
    for (float k : moneyness) {
        if (evaluate_variance_second_derivative(k, params) < 0) {
            return false;
        }
    }
    return true;
}

bool Fitter::fit_slice_scalar(
    const std::vector<float>& moneyness,
    const std::vector<float>& ivs,
    Params& params,
    float time_budget_micros
) {
    if (moneyness.size() != ivs.size() || moneyness.size() < 2) {
        return false;  // Allow fitting with 2+ data points
    }
    
    initial_guess(moneyness, ivs, params);
    bool success = optimize_scalar(moneyness, ivs, params, time_budget_micros);
    
    // Multiple recovery strategies if arbitrage check fails
    if (success && !is_butterfly_arb_free(params, moneyness)) {
        // Strategy 1: Increase sigma (smoothness)
        params.sigma *= 1.3f;
        enforce_bounds(params);
        
        // Strategy 2: If still failing, reduce rho and increase sigma more
        if (!is_butterfly_arb_free(params, moneyness)) {
            params.rho *= 0.5f;
            params.sigma *= 1.2f;
            enforce_bounds(params);
        }
        
        // Strategy 3: Last resort - conservative parameters
        if (!is_butterfly_arb_free(params, moneyness)) {
            params.rho = 0.0f;  // No skew
            params.b = std::min(params.b, 0.5f);  // Conservative slope
            params.sigma = std::max(params.sigma, 0.2f);  // Ensure smoothness
            enforce_bounds(params);
        }
    }
    
    return success;
}

#if defined(__x86_64__)
void evaluate_iv_avx512(
    const float* k_array,
    const Params& params,
    float* iv_array,
    int n
) {
    const __m512 a_vec = _mm512_set1_ps(params.a);
    const __m512 b_vec = _mm512_set1_ps(params.b);
    const __m512 rho_vec = _mm512_set1_ps(params.rho);
    const __m512 m_vec = _mm512_set1_ps(params.m);
    const __m512 sigma_sq_vec = _mm512_set1_ps(params.sigma * params.sigma);
    const __m512 min_var = _mm512_set1_ps(0.0001f);

    for (int i = 0; i < n; i += 16) {
        int remaining = std::min(16, n - i);
        __mmask16 mask = (1 << remaining) - 1;
        
        __m512 k_vec = _mm512_maskz_loadu_ps(mask, &k_array[i]);
        __m512 k_m = _mm512_sub_ps(k_vec, m_vec);
        __m512 k_m_sq = _mm512_mul_ps(k_m, k_m);
        __m512 sqrt_term = _mm512_sqrt_ps(_mm512_add_ps(k_m_sq, sigma_sq_vec));
        __m512 rho_k_m = _mm512_mul_ps(rho_vec, k_m);
        __m512 inner = _mm512_add_ps(rho_k_m, sqrt_term);
        __m512 variance = _mm512_add_ps(a_vec, _mm512_mul_ps(b_vec, inner));
        
        // Ensure positive variance and take square root for IV
        variance = _mm512_max_ps(variance, min_var);
        __m512 iv = _mm512_sqrt_ps(variance);
        
        _mm512_mask_storeu_ps(&iv_array[i], mask, iv);
    }
}

float Fitter::objective_avx512(
    const Params& params,
    const std::vector<float>& moneyness,
    const std::vector<float>& ivs
) {
    int n = moneyness.size();
    
    // Allocate aligned arrays for vectorized computation
    alignas(64) float k_array[n];
    alignas(64) float model_iv_array[n];
    
    // Copy moneyness to aligned array
    for (int i = 0; i < n; ++i) {
        k_array[i] = moneyness[i];
    }
    
    evaluate_iv_avx512(k_array, params, model_iv_array, n);
    
    __m512 sum_vec = _mm512_setzero_ps();
    for (int i = 0; i < n; i += 16) {
        __m512 model_vec = _mm512_loadu_ps(&model_iv_array[i]);
        __m512 market_vec = _mm512_loadu_ps(&ivs[i]);  
        __m512 diff = _mm512_sub_ps(model_vec, market_vec);
        __m512 sq_error = _mm512_mul_ps(diff, diff);
        sum_vec = _mm512_add_ps(sum_vec, sq_error);
    }
    float sum_sq_error = _mm512_reduce_add_ps(sum_vec);
    
    return sum_sq_error / n;
}

bool Fitter::optimize_avx512(
    const std::vector<float>& moneyness,
    const std::vector<float>& ivs,
    Params& params,
    float time_budget_micros
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    float learning_rate = 0.01f;
    int max_iterations = 150;  // More iterations for difficult cases
    float tolerance = std::max(1e-5f, prev_obj * 0.01f);  // Adaptive tolerance based on initial error
    
    float prev_obj = objective_avx512(params, moneyness, ivs);
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        auto now = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
        if (elapsed >= time_budget_micros) {
            break;
        }
        
        // Compute gradients using finite differences with AVX-512 objective
        const float h = 1e-4f;
        Params temp_params = params;
        
        temp_params.a += h;
        float grad_a = (objective_avx512(temp_params, moneyness, ivs) - prev_obj) / h;
        
        temp_params = params;
        temp_params.b += h;
        float grad_b = (objective_avx512(temp_params, moneyness, ivs) - prev_obj) / h;
        
        temp_params = params;
        temp_params.rho += h;
        float grad_rho = (objective_avx512(temp_params, moneyness, ivs) - prev_obj) / h;
        
        temp_params = params;
        temp_params.m += h;
        float grad_m = (objective_avx512(temp_params, moneyness, ivs) - prev_obj) / h;
        
        temp_params = params;
        temp_params.sigma += h;
        float grad_sigma = (objective_avx512(temp_params, moneyness, ivs) - prev_obj) / h;
        
        // Update parameters
        Params new_params = params;
        new_params.a -= learning_rate * grad_a;
        new_params.b -= learning_rate * grad_b;
        new_params.rho -= learning_rate * grad_rho;
        new_params.m -= learning_rate * grad_m;
        new_params.sigma -= learning_rate * grad_sigma;
        
        enforce_bounds(new_params);
        
        float new_obj = objective_avx512(new_params, moneyness, ivs);
        
        if (new_obj < prev_obj) {
            params = new_params;
            learning_rate *= 1.1f;
            
            if (std::abs(new_obj - prev_obj) < tolerance) {
                return true;
            }
            prev_obj = new_obj;
        } else {
            learning_rate *= 0.7f;
            if (learning_rate < 1e-6f) {
                break;
            }
        }
    }
    
    return prev_obj < 0.1f; 
}

bool Fitter::fit_slice_avx512(
    const std::vector<float>& moneyness,
    const std::vector<float>& ivs,
    Params& params,
    float time_budget_micros
) {
    if (moneyness.size() != ivs.size() || moneyness.size() < 2) {
        return false;  // Allow fitting with 2+ data points
    }
    
    initial_guess(moneyness, ivs, params);
    bool success = optimize_avx512(moneyness, ivs, params, time_budget_micros);
    
    // Multiple recovery strategies if arbitrage check fails
    if (success && !is_butterfly_arb_free(params, moneyness)) {
        // Strategy 1: Increase sigma (smoothness)
        params.sigma *= 1.3f;
        enforce_bounds(params);
        
        // Strategy 2: If still failing, reduce rho and increase sigma more
        if (!is_butterfly_arb_free(params, moneyness)) {
            params.rho *= 0.5f;
            params.sigma *= 1.2f;
            enforce_bounds(params);
        }
        
        // Strategy 3: Last resort - conservative parameters
        if (!is_butterfly_arb_free(params, moneyness)) {
            params.rho = 0.0f;  // No skew
            params.b = std::min(params.b, 0.5f);  // Conservative slope
            params.sigma = std::max(params.sigma, 0.2f);  // Ensure smoothness
            enforce_bounds(params);
        }
    }
    
    return success;
}
#endif

bool Fitter::fit_slice(
    const std::vector<float>& moneyness,
    const std::vector<float>& ivs,
    Params& params,
    float time_budget_micros
) {
    if (moneyness.size() != ivs.size() || moneyness.size() < 2) {
        return false;  // Allow fitting with 2+ data points
    }
    
#if defined(__x86_64__)
    #ifdef HAS_AVX512
    return fit_slice_avx512(moneyness, ivs, params, time_budget_micros);
    #else
    return fit_slice_scalar(moneyness, ivs, params, time_budget_micros);
    #endif
#else
    return fit_slice_scalar(moneyness, ivs, params, time_budget_micros);
#endif
}

} // namespace SVI
} // namespace SurfaceFitter
