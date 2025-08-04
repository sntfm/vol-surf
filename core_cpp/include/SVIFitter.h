#pragma once

#include <vector>
#include "flatbuffers/flatbuffers.h"

namespace svi {

struct SVIParams {
    float a;     // overall level
    float b;     // slope of wings
    float rho;   // correlation (-1 to 1)
    float m;     // shift
    float sigma; // smoothness

    // Butterfly arbitrage condition: w''(k) ≥ 0
    bool is_butterfly_arbitrage_free(float k) const;
    
    // Calendar arbitrage condition: ∂w/∂T ≥ 0
    // Note: This requires term structure fitting
    bool is_calendar_arbitrage_free(float k, float T1, float T2) const;
};

class SVIFitter {
public:
    // Fit SVI parameters to a slice of implied volatilities
    // Returns true if fit successful within time budget
    bool fit_slice(
        const flatbuffers::Vector<float>* moneyness,  // k = log(K/F)
        const flatbuffers::Vector<float>* ivs,
        SVIParams& params,
        float time_budget_micros = 200.0f
    );

    // Direct vector interface - faster for performance-critical applications
    bool fit_slice_direct(
        const std::vector<float>& moneyness,  // k = log(K/F)
        const std::vector<float>& ivs,
        SVIParams& params,
        float time_budget_micros = 200.0f
    );

    // Scalar implementation - maximum compatibility
    bool fit_slice_direct_scalar(
        const std::vector<float>& moneyness,
        const std::vector<float>& ivs,
        SVIParams& params,
        float time_budget_micros = 200.0f
    );

    // AVX2 implementation - good balance of speed and compatibility
    bool fit_slice_direct_avx2(
        const std::vector<float>& moneyness,
        const std::vector<float>& ivs,
        SVIParams& params,
        float time_budget_micros = 200.0f
    );

    // AVX-512 implementation - maximum performance
    bool fit_slice_direct_avx512(
        const std::vector<float>& moneyness,
        const std::vector<float>& ivs,
        SVIParams& params,
        float time_budget_micros = 200.0f
    );

    // Auto-select best implementation based on CPU capabilities
    static bool detect_best_implementation();

    // Evaluate SVI variance at given moneyness
    static float evaluate(float k, const SVIParams& params);
    
    // Evaluate first derivative of total variance
    static float evaluate_first_derivative(float k, const SVIParams& params);
    
    // Evaluate second derivative of total variance
    static float evaluate_second_derivative(float k, const SVIParams& params);

private:
    // Initial guess based on raw moments
    void initial_guess(
        const flatbuffers::Vector<float>* moneyness,
        const flatbuffers::Vector<float>* ivs,
        SVIParams& params
    );

    // Objective function for optimization
    float objective(
        const SVIParams& params,
        const flatbuffers::Vector<float>* moneyness,
        const flatbuffers::Vector<float>* ivs
    );

    // Gradient of objective function
    void gradient(
        const SVIParams& params,
        const flatbuffers::Vector<float>* moneyness,
        const flatbuffers::Vector<float>* ivs,
        float grad[5]
    );

    // Levenberg-Marquardt step with arbitrage constraints
    bool lm_step(
        SVIParams& params,
        const flatbuffers::Vector<float>* moneyness,
        const flatbuffers::Vector<float>* ivs,
        float lambda
    );

    // Parameter bounds check and projection
    static void enforce_bounds(SVIParams& params);
    
    // Check and enforce arbitrage-free conditions
    static void enforce_arbitrage_free(
        SVIParams& params,
        const flatbuffers::Vector<float>* moneyness
    );
};

} // namespace svi 