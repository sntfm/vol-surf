#pragma once

#include <vector>

namespace SurfaceFitter {
namespace SVI {

struct Params {
    float a;     // level
    float b;     // angle  
    float rho;   // correlation (-1 to 1)
    float m;     // translation
    float sigma; // smoothness
    
    Params() : a(0.1f), b(0.5f), rho(0.0f), m(0.0f), sigma(0.2f) {}
};

class Fitter {
public:
    // Selects implementation at compile time
    bool fit_slice(
        const std::vector<float>& moneyness,  // k = log(K/F)
        const std::vector<float>& ivs,        // implied volatilities
        Params& params,
        float time_budget_micros = 1000.0f
    );

private:
    // Initial parameter guess from data moments
    void initial_guess(
        const std::vector<float>& moneyness,
        const std::vector<float>& ivs,
        Params& params
    );
    
    // Evaluate SVI total variance at given moneyness: w(k) = a + b * (rho * (k-m) + sqrt((k-m)^2 + sigma^2))
    static float evaluate_variance(float k, const Params& params);
    
    // Evaluate SVI implied volatility at given moneyness
    static float evaluate_iv(float k, const Params& params);
    
    // Evaluate second derivative of total variance: d2w/dk2
    static float evaluate_variance_second_derivative(float k, const Params& params);
    
    // Check butterfly arbitrage condition: d2w/dk2 ≥ 0 for all strikes
    static bool is_butterfly_arb_free(const Params& params, const std::vector<float>& moneyness);
    
    // Enforce parameter bounds
    static void enforce_bounds(Params& params);

    // Scalar implementation
    bool fit_slice_scalar(
        const std::vector<float>& moneyness,  // k = log(K/F)
        const std::vector<float>& ivs,        // implied volatilities
        Params& params,
        float time_budget_micros = 1000.0f
    );

    float objective_scalar(
        const Params& params,
        const std::vector<float>& moneyness,
        const std::vector<float>& ivs
    );

    // Simple gradient descent optimization
    bool optimize_scalar(
        const std::vector<float>& moneyness,
        const std::vector<float>& ivs,
        Params& params,
        float time_budget_micros
    );

#if defined(__x86_64__)
    // AVX-512 implementation
    bool fit_slice_avx512(
        const std::vector<float>& moneyness,  // k = log(K/F)
        const std::vector<float>& ivs,        
        Params& params,
        float time_budget_micros = 1000.0f
    );

    float objective_avx512(
        const Params& params,
        const std::vector<float>& moneyness,
        const std::vector<float>& ivs
    );

    bool optimize_avx512(
        const std::vector<float>& moneyness,
        const std::vector<float>& ivs,
        Params& params,
        float time_budget_micros
    );
#endif
};

} // namespace SVI
} // namespace SurfaceFitter